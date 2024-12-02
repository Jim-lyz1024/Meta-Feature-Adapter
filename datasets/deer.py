import glob
import re

import os.path as osp

import json

import os

from .bases import BaseImageDataset
from collections import defaultdict
import pickle


class DEER(BaseImageDataset):
    """
    Deer Dataset
    """
    dataset_dir = "Deer"

    def __init__(self, root='', verbose=True, pid_begin=0, data_p='/data/yil708/Code-CLIP-ReID/datasets_meta/deer.json', **kwargs):
        super(DEER, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        with open(data_p, 'rb') as f:
            data = json.load(f)

        infos = {}
        for d in data['images']:
            # Store both lowercase and original versions
            img_name = d['img_path'].split('\\')[-1]
            infos[img_name.lower()] = d['metadata']
            infos[img_name] = d['metadata']

        self.infos = infos

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Deer loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    def get_metalabel(self, dataset_info):

        temperature = dataset_info['temperature']
        humidity = 80
        light = dataset_info['day_night']
        angle = dataset_info['face_direction']

        temperature = float(temperature)
        humidity = float(humidity)
        angle = float(angle)
        light = float(light)

        if temperature < 17:
            temperature_label = 'cold'
        elif temperature >= 17 and temperature < 20:
            temperature_label = 'mild'
        elif temperature >= 20:
            temperature_label = 'hot'

        if humidity < 75:
            humidity_label = 'dry'
        elif humidity >= 75 and humidity < 85:
            humidity_label = 'moderate'
        elif humidity >= 85:
            humidity_label = 'humid'

        # if rain == 0:
        #     rain_label = 'no rain'
        # elif rain > 0 and rain <= 10:
        #     rain_label = 'light'
        # elif rain > 10 and rain <= 50:
        #     rain_label = 'moderate'
        # elif rain >= 50:
        #     rain_label = 'heavy'
        
        if light == 0:
            light_label = 'day'
        elif light == 1:
            light_label = 'night'

        if angle==0:
            angle_label='front'
        elif angle==1:
            angle_label='back'
        elif angle==2:
            angle_label='left'
        elif angle==3:
            angle_label='right'

        print(temperature_label, humidity_label, light_label, angle_label)
        # exit()
        return temperature_label, humidity_label, light_label, angle_label

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.extend(glob.glob(osp.join(dir_path, '*.JPG')))  # Handle both upper and lower case extensions
        
        pattern = re.compile(r'(\d+)_([A-Za-z0-9-]+)_(\d+)')
        
        print(f"\nProcessing directory: {dir_path}")
        print(f"Number of images found: {len(img_paths)}")

        pid_container = set()
        camid_container = set()
        
        # First pass to collect PIDs and camera IDs
        for img_path in sorted(img_paths):
            basename = osp.basename(img_path)
            match = pattern.match(basename)
            if match:
                pid = int(match.group(1))
                camid = hash(match.group(2)) % 10000  # Convert camera string to numeric ID
                pid_container.add(pid)
                camid_container.add(camid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Second pass to build dataset with metadata
        dataset = []
        for img_path in sorted(img_paths):
            basename = osp.basename(img_path)
            match = pattern.match(basename)
            if match:
                pid = int(match.group(1))
                camid = hash(match.group(2)) % 10000
                
                if relabel:
                    pid = pid2label[pid]
                
                # Try to find metadata using case-insensitive lookup
                try:
                    dataset_info = self.infos.get(basename) or self.infos.get(basename.lower())
                    if dataset_info is None:
                        print(f"Warning: No metadata found for {basename}")
                        continue
                    metalabel = self.get_metalabel(dataset_info)
                    dataset.append((img_path, self.pid_begin + pid, camid, 0, *metalabel))
                except KeyError as e:
                    print(f"Warning: Failed to process metadata for {basename}: {e}")
                    continue

        return dataset

