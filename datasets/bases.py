from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(f"IOError incurred when reading '{img_path}'. Will redo.")
            pass
    return img

def read_metadata(json_path):
    """Read metadata from JSON file"""
    if not osp.exists(json_path):
        raise IOError(f"Metadata file {json_path} does not exist")
    with open(json_path, 'r') as f:
        metadata_dict = json.load(f)
    return metadata_dict

class BaseDataset(object):
    """Base class of reid dataset"""
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        # Updated to handle 5 values (img_path, pid, camid, trackid, img_name)
        for img_path, pid, camid, trackid, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError

class BaseImageDataset(BaseDataset):
    """Base class of image reid dataset"""
    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print(" ----------------------------------------")
        print(" subset | # ids | # images | # cameras")
        print(" ----------------------------------------")
        print(f" train | {num_train_pids:5d} | {num_train_imgs:8d} | {num_train_cams:9d}")
        print(f" query | {num_query_pids:5d} | {num_query_imgs:8d} | {num_query_cams:9d}")
        print(f" gallery | {num_gallery_pids:5d} | {num_gallery_imgs:8d} | {num_gallery_cams:9d}")
        print(" ----------------------------------------")

class ImageDataset(Dataset):
    """Image dataset with metadata support"""
    def __init__(self, dataset, transform=None, metadata_path=None):
        self.dataset = dataset
        self.transform = transform
        self.metadata = {}

        # Load metadata if provided
        if metadata_path is not None:
            self.metadata = read_metadata(metadata_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, img_name = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # Get metadata if available
        metadata = None
        if self.metadata:
            img_key = img_name.split('/')[-1]  # Get filename
            for img_data in self.metadata['images']:
                if img_data['img_path'].replace('\\','/').endswith(img_key):
                    metadata = {
                        'temperature': float(img_data['metadata']['temperature']),
                        'humidity': float(img_data['metadata']['humidity']),
                        'rain': float(img_data['metadata']['rain']),
                        'angle': int(img_data['metadata']['angle'])
                    }
                    break

        if metadata is None:
            metadata = {
                'temperature': 20.0,  # Default values
                'humidity': 80.0,
                'rain': 0.0,
                'angle': 0
            }

        return img, pid, camid, trackid, img_name, metadata