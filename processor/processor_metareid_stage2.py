import logging
import os
import time
from datetime import timedelta
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage2(cfg,
                   model,
                   center_criterion,
                   train_loader_stage2,
                   val_loader,
                   optimizer,
                   optimizer_center,
                   scheduler,
                   loss_fn,
                   num_query, local_rank):
    """
    Stage 2 training: Train full model with metadata enhancement
    """
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('Start training stage 2 - full model with metadata')

    # Record total training time
    all_start_time = time.monotonic()

    # Setup models
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    # Initialize meters and evaluator
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, dataset_name=cfg.DATASETS.NAMES, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # Generate text features for all IDs
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes//batch)
    if left != 0:
        i_ter = i_ter + 1

    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
                text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    # Training loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()

        for n_iter, (img, vid, target_cam, target_view, _, metadata) in enumerate(train_loader_stage2):
            print(f"cam_label: {target_cam}")
            print(f"view_label: {target_view}")
            # exit()
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # Move data to GPU
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            
            # Move metadata to GPU
            metadata = {k: v.to(device) for k, v in metadata.items()}

            with amp.autocast(enabled=True):
                # Forward pass with metadata
                score, feat, image_features = model(
                    x=img,
                    metadata=metadata, 
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view
                )
                
                # Calculate ID-Text matching loss
                logits = image_features @ text_features.t()

                # Calculate losses including logits
                loss = loss_fn(score, feat, target, target_cam, logits)

                # Update model
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update center loss if used
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()

            # Update metrics
            acc = (score[0].max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            # Log progress
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                          .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        # Compute epoch time
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info(
            f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s] "
            f"Speed: {train_loader_stage2.batch_size / time_per_batch:.1f}[samples/s]"
        )

        # Save checkpoint
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0:
                torch.save(model.state_dict(),
                         os.path.join(cfg.OUTPUT_DIR, f'{cfg.MODEL.NAME}_meta_{epoch}.pth'))
            elif not cfg.MODEL.DIST_TRAIN:
                torch.save(model.state_dict(),
                         os.path.join(cfg.OUTPUT_DIR, f'{cfg.MODEL.NAME}_meta_{epoch}.pth'))

        # Validation
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _, metadata) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
                    target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
                    metadata = {k: v.to(device) for k, v in metadata.items()}
                    feat = model(img, metadata=metadata, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))

            # Calculate metrics
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

    # Record and log total training time
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f"Total training time: {total_time}")

def do_inference(cfg,
                model,
                val_loader,
                num_query):
    """
    Inference function with metadata support
    """
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, dataset_name=cfg.DATASETS.NAMES, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs for inference')
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, metadata) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            metadata = {k: v.to(device) for k, v in metadata.items()}
            feat = model(img, metadata=metadata, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    # Calculate metrics
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return cmc[0], cmc[4]