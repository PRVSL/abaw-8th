"""
Author: Huynh Van Thong
https://pr.ai.vn
"""

import math
import pathlib

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils import data

from PIL import Image
from torchvision.transforms import v2


class ABAWDataset(data.Dataset):
    def __init__(self, data_dir, task, seq_len, split='Train', transforms=None, fold=None,
                 sampling_method='sequentially'):
        self.root_dir = data_dir
        self.task = task
        self.split = split
        self.seq_len = seq_len
        self.sampling_method = sampling_method
        self.transforms = transforms

        if split != 'Test':
            data_dict = \
                np.load(pathlib.Path(self.root_dir, '{}.npy'.format(self.task)).__str__(), allow_pickle=True).item()[
                    self.split]
        else:
            test_df = pd.read_csv(
                pathlib.Path(self.root_dir, 'test_set/CVPR_8th_ABAW_{}_test_set_example.txt'.format(self.task)),
                sep=',', header=0)
            test_df['video_name'] = test_df['image_location'].apply(lambda x: x.split('/')[0])
            data_dict = dict()
            for video_name, gp in test_df.groupby('video_name', sort=False):
                data_dict[video_name] = gp.drop('video_name', axis=1).values

        self.data_seqs = []

        for vid in data_dict:
            if fold is not None:
                if vid.replace('_left', '').replace('_right', '') not in fold:
                    continue
            cur_vid_df = data_dict[vid]
            num_frames = cur_vid_df.shape[0]
            overlap = int(1.*self.seq_len) if self.split != 'Test' else int(0.5*self.seq_len)
            num_seqs = math.ceil(num_frames / overlap) #math.ceil(num_frames / self.seq_len)

            array_indexes = np.arange(num_frames)
            cur_set = set(array_indexes.flatten())

            # Count the number of sequences
            for idx in range(num_seqs):
                if self.sampling_method == 'sequentially' or self.seq_len == 1:
                    st_idx = idx * overlap # self.seq_len
                    ed_idx = min(st_idx + self.seq_len, num_frames)# min((idx + 1) * self.seq_len, num_frames)
                    # Get the sequence
                    cur_seq = cur_vid_df[st_idx: ed_idx, :]
                    # if self.task == 'EXPR' and cur_seq[0][1] in ['0', '7'] and torch.rand(1) < 0.5 and self.split == 'Train':
                    #     continue
                elif self.sampling_method == 'randomly':
                    if len(cur_set) > self.seq_len:
                        cur_idx = np.random.choice(np.array(list(cur_set)), self.seq_len, replace=False)
                    else:
                        cur_idx = np.array(list(cur_set))

                    cur_idx.sort()

                    cur_seq = cur_vid_df[cur_idx, :]
                    cur_set = cur_set.difference(set(cur_idx.flatten()))
                else:
                    raise ValueError('Only support sequentially or random at this time.')

                # Padding if need
                if cur_seq.shape[0] < self.seq_len:
                    # Do Padding, Pad to the end of the sequence by copying
                    cur_seq = np.pad(cur_seq, ((0, self.seq_len - cur_seq.shape[0]), (0, 0)), 'edge')

                self.data_seqs.append(cur_seq)

    def __len__(self):
        return len(self.data_seqs)

    def get_labels(self):
        if self.task == 'EXPR':
            list_of_label = [x[:, 1].astype(np.int) for x in self.data_seqs]
            list_of_label = np.concatenate(list_of_label).flatten()
            return list_of_label

        elif self.task == 'VA':
            list_of_label = [x[:, 2].astype(float) for x in self.data_seqs]  # Calculate base on valence
            list_of_label = np.concatenate(list_of_label).flatten()
            bins = [-1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.1]
            list_of_label = np.digitize(list_of_label, bins, right=False) - 1
            return list_of_label
        else:
            raise ValueError('get_label() method was not implemented for {}.'.format(self.task))
        pass

    def __getitem__(self, index):
        """
        Task VA: file_name, Valence, Arousal, Frame Index (Total 4 columns)
        Task EXPR: file_name, Emotion index (0,1,...,7), Frame index (Total 3 columns)
        Task AU: file_name, 12 action unit index (0, 1), Frame index (multi-label classification (Total 14 columns))
        :param index:
        :return:
        """

        cur_seq_sample = self.data_seqs[index]

        sample = {'image': [], #'index': cur_seq_sample[:, -1].astype(np.int32),
                  'video_id': cur_seq_sample[:, 0].tolist()}  #.split('/')[0]
        sample.update({self.task: []})

        for idx in range(cur_seq_sample.shape[0]):
            # Load image
            cur_img_path = pathlib.Path(self.root_dir, 'cropped_aligned', cur_seq_sample[idx, 0])
            if cur_img_path.is_file():
                cur_img = Image.open(cur_img_path).convert('RGB')
            else:
                cur_img = Image.new('RGB', (112, 112))

            if self.transforms is not None:
                cur_img = self.transforms(cur_img)

            sample['image'].append(cur_img)

            if self.task == 'VA':
                sample['VA'].append([float(x) for x in cur_seq_sample[idx, 1:3]])
            elif self.task in ['EXPR', 'AU']:
                sample[self.task].append([int(x) for x in cur_seq_sample[idx, 1:-1]])
            else:
                raise ValueError(f'Un-supported {self.task} task')

        sample['image'] = torch.stack(sample['image'], dim=0)
        for ky in ['VA', 'EXPR', 'AU']:
            if ky in sample.keys():
                sample[ky] = np.array(sample[ky])
                sample[ky] = torch.from_numpy(sample[ky])

        return sample


class ABAWDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, task, seq_len, img_size=112, num_folds=0, num_workers=4, batch_size=32):
        super(ABAWDataModule, self).__init__()

        self.data_dir = data_dir
        self.task = task
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_workers = num_workers

        aug = [v2.RandomHorizontalFlip(p=0.5),
               v2.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25),
               v2.RandomErasing(scale=(0.1, 0.1), ratio=(0.7, 1.3)),
               v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)), ]

        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        train_augs = [v2.ToDtype(torch.float32, scale=True), normalize]
        val_augs = [v2.ToDtype(torch.float32, scale=True), normalize]

        if img_size < 112:
            train_augs = [v2.RandomCrop(img_size), ] + train_augs
            val_augs = [v2.CenterCrop(img_size), ] + val_augs
        elif img_size > 112:
            train_augs = [v2.Resize(img_size, antialias=None), ] + train_augs
            val_augs = [v2.Resize(img_size, antialias=None), ] + val_augs
        else:
            pass

        self.transforms_train = v2.Compose([v2.ToImage(), ] + aug + train_augs)
        self.transforms_test = v2.Compose([v2.ToImage(), ] + val_augs)

        self.num_folds = num_folds
        self.train_fold = None
        self.val_fold = None
        self.is_setup_folds = False

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Create train set and val set
            self.train_dataset = ABAWDataset(self.data_dir, self.task, self.seq_len, split='Train',
                                             transforms=self.transforms_train)
            self.val_dataset = ABAWDataset(self.data_dir, self.task, self.seq_len, split='Validation',
                                           transforms=self.transforms_test)

        if stage == 'test' or stage is None:
            # Create test set
            self.test_dataset = ABAWDataset(self.data_dir, self.task, self.seq_len, split='Test',
                                            transforms=self.transforms_test)

    def train_dataloader(self, shufflex=None):
        sampler = None
        shuffle = True
        if shufflex is not None:
            shuffle = shufflex

        if self.train_fold is not None:
            train_dataset = ABAWDataset(split='Train', transforms=self.transforms_train, fold=self.train_fold)
        else:
            train_dataset = self.train_dataset

        return data.DataLoader(train_dataset, batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=shuffle, sampler=sampler, prefetch_factor=4)

    def val_dataloader(self):
        if self.val_fold is not None:
            val_dataset = ABAWDataset(split='Train', transforms=self.transforms_test, fold=self.val_fold)
        else:
            val_dataset = self.val_dataset

        return data.DataLoader(val_dataset, batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=False)
