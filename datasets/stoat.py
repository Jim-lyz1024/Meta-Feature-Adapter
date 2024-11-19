import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import json

class STOAT(BaseImageDataset):
    """
    New Zealand (Waiheke Island and South Island) Stoat Dataset
    
    Dataset statistics:
    # train - South Island, gallery - Waiheke Island, query - Waiheke Island
    # identities: 56 (train) + 5 (gallery) + 5 (query)
    # images: 183 (train) + 13 (gallery) + 13 (query)
    """
    dataset_dir = "Stoat"

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(STOAT, self).__init__()
        
        # Set paths
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        
        # Set metadata path
        self.metadata_file = "/data/yil708/Code-CLIP-ReID/Meta-Feature-Adapter/data/stoat.json"
        
        self._check_before_run()
        self.pid_begin = pid_begin
        
        # Process directories
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        if verbose:
            print("=> Stoat loaded")
            self.print_dataset_statistics(train, query, gallery)
            
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")
        if not osp.exists(self.metadata_file):
            raise RuntimeError(f"Metadata file '{self.metadata_file}' is not available")
            
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        # Handle special camera names
        special_cameras = {
            "CREK": 1000,
            "FC01": 1001,
            "FC11": 1002,
            "GC34": 1003,
            "P164": 1004
        }
        
        pattern = re.compile(r'\d+_[0-9a-zA-Z]+_\d+')
        pid_container = set()
        camid_container = set()
        for img_path in sorted(img_paths):
            pid, camid, _ = pattern.search(img_path).group().split("_")
            pid = int(pid)
            if camid in special_cameras:
                camid = special_cameras[camid]
            camid = int(camid)
            pid_container.add(pid)
            camid_container.add(camid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid, _ = pattern.search(img_path).group().split("_")
            pid = int(pid)
            if camid in special_cameras:
                camid = special_cameras[camid]
            camid = int(camid)
            assert 0 <= pid <= 55
            if relabel:
                pid = pid2label[pid]
            # Store full relative path as img_name for metadata lookup
            img_name = img_path.split('data/Stoat/')[-1] if 'data/Stoat/' in img_path else img_path
            dataset.append((img_path, self.pid_begin + pid, camid, 0, img_name))
        return dataset