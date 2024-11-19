import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .stoat import STOAT
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

__factory = {
    'stoat': STOAT,
}

def train_collate_fn(batch):
    """
    Collate function for training data
    Args:
        batch: list of tuples (image, pid, camid, trackid, img_name, metadata)
    """
    imgs, pids, camids, trackids, img_names, metadata_list = zip(*batch)
    
    # Convert PIL images to tensor batch
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    trackids = torch.tensor(trackids, dtype=torch.int64)
    
    # Convert metadata list to dictionary of batched values
    batch_metadata = {
        'temperature': torch.tensor([m['temperature'] for m in metadata_list], dtype=torch.float32),
        'humidity': torch.tensor([m['humidity'] for m in metadata_list], dtype=torch.float32),
        'rain': torch.tensor([m['rain'] for m in metadata_list], dtype=torch.float32),
        'angle': torch.tensor([m['angle'] for m in metadata_list], dtype=torch.int64)
    }
    
    return imgs, pids, camids, trackids, img_names, batch_metadata

def val_collate_fn(batch):
    """
    Collate function for validation/test data
    Args:
        batch: list of tuples (image, pid, camid, trackid, img_name, metadata)
    """
    imgs, pids, camids, trackids, img_names, metadata_list = zip(*batch)
    
    imgs = torch.stack(imgs, dim=0)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    trackids = torch.tensor(trackids, dtype=torch.int64)
    
    batch_metadata = {
        'temperature': torch.tensor([m['temperature'] for m in metadata_list], dtype=torch.float32),
        'humidity': torch.tensor([m['humidity'] for m in metadata_list], dtype=torch.float32),
        'rain': torch.tensor([m['rain'] for m in metadata_list], dtype=torch.float32),
        'angle': torch.tensor([m['angle'] for m in metadata_list], dtype=torch.int64)
    }
    
    return imgs, pids, camids, camids_batch, trackids, img_names, batch_metadata

def make_dataloader(cfg):
    """
    Create dataloaders for training and testing
    Args:
        cfg: config object containing dataset and dataloader parameters
    """
    # Define data transforms
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # Initialize dataset
    metadata_path = "/data/yil708/Code-CLIP-ReID/Meta-Feature-Adapter/data/stoat.json"
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    # Create dataset instances
    train_set = ImageDataset(dataset.train, train_transforms, metadata_path=metadata_path)
    train_set_normal = ImageDataset(dataset.train, val_transforms, metadata_path=metadata_path)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, metadata_path=metadata_path)

    # Get dataset information
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # Create data loaders
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                dataset.train,
                cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler,
                mini_batch_size,
                True
            )
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    dataset.train,
                    cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE
                ),
                num_workers=num_workers,
                collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('Using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print(f'Unsupported sampler! Expected softmax or triplet but got {cfg.SAMPLER}')
        raise ValueError

    # Create validation loader
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # Create stage1 training loader
    train_loader_stage1 = DataLoader(
        train_set_normal,
        batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num