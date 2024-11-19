# import os
# import sys
# # Add the project root directory to the system path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from datasets.make_dataloader_metareid import make_dataloader
from model.make_model_metareid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_metareid_stage1 import do_train_stage1
from processor.processor_metareid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MetaReID Training")
    parser.add_argument(
        "--config_file", 
        default="configs/animal/vit_metareid.yml", 
        help="path to config file", 
        type=str
    )
    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line", 
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Set random seed
    set_seed(cfg.SOLVER.SEED)

    # Set device
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    # Create output directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logger
    logger = setup_logger("metareid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Initialize distributed training if enabled
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Create data loaders
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Create model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Create loss function
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # Stage 1: Train metadata text generator
    logger.info("Starting Stage 1 - Training metadata text generator")
    
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(
        optimizer_1stage, 
        num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
        lr_min=cfg.SOLVER.STAGE1.LR_MIN,
        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
        noise_range=None
    )
    
    print("Starting Stage 1 training-------------------")

    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    # Stage 2: Train full model with metadata
    logger.info("Starting Stage 2 - Training full model with metadata")
    
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage, 
        cfg.SOLVER.STAGE2.STEPS, 
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD
    )

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, 
        args.local_rank
    )

    logger.info("Training completed. Model saved to: {}".format(cfg.OUTPUT_DIR))