import os
from config import cfg
import argparse
from datasets.make_dataloader_metareid import make_dataloader
from model.make_model_metareid import make_model
from processor.processor_metareid_stage2 import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaReID Testing")
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
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create output directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logger    
    logger = setup_logger("metareid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Set visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Create data loaders
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Create model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Load trained weights
    model.load_param(cfg.TEST.WEIGHT)

    # Test on VehicleID dataset
    if cfg.DATASETS.NAMES == 'VehicleID':
        # Run multiple trials for VehicleID dataset
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5, mAP = do_inference(cfg,
                                            model,
                                            val_loader,
                                            num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_mAP = mAP
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_mAP = all_mAP + mAP

            logger.info("Trial {}: rank-1: {:.1%}, rank-5 {:.1%}, mAP {:.1%}"
                       .format(trial, rank_1, rank5, mAP))

        # Print average results
        logger.info("Average Results:")
        logger.info("Rank-1: {:.1%}".format(all_rank_1.sum()/10.0))
        logger.info("Rank-5: {:.1%}".format(all_rank_5.sum()/10.0))
        logger.info("mAP: {:.1%}".format(all_mAP.sum()/10.0))
    else:
        # Test on other datasets
        do_inference(cfg,
                    model,
                    val_loader,
                    num_query)