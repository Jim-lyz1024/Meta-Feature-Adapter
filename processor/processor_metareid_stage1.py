import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import time
from datetime import timedelta

def do_train_stage1(cfg,
                   model,
                   train_loader_stage1,
                   optimizer,
                   scheduler,
                   local_rank):
    """
    Stage 1 training: Train metadata text generator and optimize text tokens
    """
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    # Setup logger
    logger = logging.getLogger("transreid.train")
    logger.info('Start training stage 1 - metadata text generation')

    # Record total training time
    all_start_time = time.monotonic()

    # Setup models and loss
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
            model = nn.DataParallel(model)

    # Initialize meters and loss
    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # Store image features
    image_features = []
    labels = []
    metadata_list = []

    # First pass: collect image features and metadata
    logger.info("Collecting image features and metadata...")
    with torch.no_grad():
        for n_iter, batch in enumerate(train_loader_stage1):
            # Unpack batch
            if len(batch) == 6:
                img, vid, _, _, _, metadata = batch
            else:
                logger.error(f"Unexpected batch format: {len(batch)} elements")
                continue

            # Debug print the first batch's metadata
            if n_iter == 0:
                logger.info(f"Sample metadata structure: {metadata}")

            img = img.to(device)
            target = vid.to(device)

            # Get image features only
            with amp.autocast(enabled=True):
                image_feature = model(img, metadata=None, get_image=True)

            # Store features and metadata
            for i in range(len(img)):
                labels.append(target[i])
                image_features.append(image_feature[i].cpu())
                # Create metadata dictionary for each instance
                instance_metadata = {
                    'temperature': metadata['temperature'][i].item(),
                    'humidity': metadata['humidity'][i].item(),
                    'rain': metadata['rain'][i].item(),
                    'angle': metadata['angle'][i].item()
                }
                metadata_list.append(instance_metadata)

        # Print some metadata examples for debugging
        logger.info(f"Sample metadata entries:")
        for i in range(min(3, len(metadata_list))):
            logger.info(f"Entry {i}: {metadata_list[i]}")

        # Convert stored data to tensors
        labels_list = torch.stack(labels, dim=0).cuda()
        image_features_list = torch.stack(image_features, dim=0).cuda()

        # Get batch size and iterations
        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch

        # Clean up memory
        del labels, image_features

    # Main training loop
    logger.info("Starting meta text generator training...")
    model.train()
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        # model.train()

        # Shuffle data order for each epoch
        iter_list = torch.randperm(num_image).to(device)

        # Mini-batch training
        for i in range(i_ter + 1):
            optimizer.zero_grad()

            # Get current batch indices
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)*batch]
            else:
                b_list = iter_list[i*batch:num_image]

            target = labels_list[b_list]
            image_features = image_features_list[b_list]

            # Process metadata for current batch
            batch_metadata = {
                'temperature': torch.tensor([metadata_list[idx.item()]['temperature'] 
                                          for idx in b_list], device=device),
                'humidity': torch.tensor([metadata_list[idx.item()]['humidity'] 
                                        for idx in b_list], device=device),
                'rain': torch.tensor([metadata_list[idx.item()]['rain'] 
                                    for idx in b_list], device=device),
                'angle': torch.tensor([metadata_list[idx.item()]['angle'] 
                                     for idx in b_list], device=device)
            }

            with amp.autocast(enabled=True):
                # Generate text features with metadata
                text_features = model(label=target, metadata=batch_metadata, get_text=True)

                # Calculate losses
                loss_i2t = xent(image_features, text_features, target, target)
                loss_t2i = xent(text_features, image_features, target, target)
                loss = loss_i2t + loss_t2i

                # Update model
                scaler.scale(loss).backward()
                print("in stage1------------")
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), len(b_list))

            # Log progress
            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                          .format(epoch, (i + 1), i_ter + 1,
                                loss_meter.avg, scheduler._get_lr(epoch)[0]))

        # Save checkpoint
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0:
                torch.save(model.state_dict(),
                         os.path.join(cfg.OUTPUT_DIR, f'{cfg.MODEL.NAME}_meta_stage1_{epoch}.pth'))
            elif not cfg.MODEL.DIST_TRAIN:
                torch.save(model.state_dict(),
                         os.path.join(cfg.OUTPUT_DIR, f'{cfg.MODEL.NAME}_meta_stage1_{epoch}.pth'))

    # Record and log total training time
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f"Stage 1 training completed in {total_time}")