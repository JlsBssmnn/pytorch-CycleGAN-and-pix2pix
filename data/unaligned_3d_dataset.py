"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>: Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from typing import Literal

import numpy as np
import torch
from sortedcontainers import SortedDict
from data.base_dataset import BaseDataset, get_transform_3d
from data.h5_folder import get_datasets
from util.logging_config import logging


class Unaligned3dDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--border_offset', type=int, nargs=3, default=[0, 0, 0], help='How much space around the z-,y- and x-borders of the image are not used for sampling images')
        parser.add_argument('--sample_size', type=int, nargs=3, required=True, help='The size along z-,y- and x-axis for a sampled image')
        parser.add_argument('--datasetA_file', type=str, required=True, help='The name of the file that contains dataset A')
        parser.add_argument('--datasetB_file', type=str, required=True, help='The name of the file that contains dataset B')
        parser.add_argument('--datasetA_names', type=str, nargs='+', help='The names of the datasets in the file that belong to dataset A')
        parser.add_argument('--datasetB_names', type=str, nargs='+', help='The names of the datasets in the file that belong to dataset B')
        parser.add_argument('--datasetA_mask', type=str, default=None, help='The name of the dataset to use as mask for dataset A')
        parser.add_argument('--datasetB_mask', type=str, default=None, help='The name of the dataset to use as mask for dataset B')
        parser.add_argument('--dataset_length', type=str, default='max', choices=['min', 'max'], help='How to determine the size of the entire dataset given the sizes of dataset A and dataset B')
        parser.add_argument('--no_normalization', action='store_true', help='If specified the samples are not normalized')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.A_images = get_datasets(os.path.join(opt.dataroot, opt.phase + 'A', opt.datasetA_file), opt.datasetA_names, [opt.datasetA_mask])
        self.B_images = get_datasets(os.path.join(opt.dataroot, opt.phase + 'B', opt.datasetB_file), opt.datasetB_names, [opt.datasetB_mask])

        offset_slicing = tuple([slice(offset, -offset if offset > 0 else None) for offset in opt.border_offset])
        self.A_images = [arr[offset_slicing] for arr in self.A_images]
        self.B_images = [arr[offset_slicing] for arr in self.B_images]

        if opt.datasetA_mask:
            assert len(self.A_images) == 1
            self.A_mask = get_datasets(os.path.join(opt.dataroot, opt.phase + 'A', opt.datasetA_file), [opt.datasetA_mask])[0][offset_slicing]
            self.init_samples_with_mask('A')
        else:
            self.init_samples_no_mask('A')
            self.A_mask = None

        if opt.datasetB_mask:
            assert len(self.B_images) == 1
            self.B_mask = get_datasets(os.path.join(opt.dataroot, opt.phase + 'B', opt.datasetB_file), [opt.datasetB_mask])[0][offset_slicing]
            self.init_samples_with_mask('B')
        else:
            self.init_samples_no_mask('B')
            self.B_mask = None
        
        assert self.samples_per_image_A[-1] > 0 and self.samples_per_image_B[-1] > 0, \
            "One of the datasets is empty (could be due to a mask)"

        if opt.dataset_length == 'min':
            self.len = int(min(self.samples_per_image_A[-1], self.samples_per_image_B[-1]))
        elif opt.dataset_length == 'max':
            self.len = int(max(self.samples_per_image_A[-1], self.samples_per_image_B[-1]))
        else:
            logging.error('Invalid dataset_length parameter!')
            exit(1)

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform_A = get_transform_3d(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform_3d(self.opt, grayscale=(output_nc == 1))

    def init_samples_no_mask(self, side: Literal['A'] | Literal['B']):
        setattr(self, 'samples_per_image_' + side, [])
        setattr(self, 'sample_shape_' + side, [])
        samples_per_image = getattr(self, 'samples_per_image_' + side)
        sample_shape = getattr(self, 'sample_shape_' + side)

        for image in getattr(self, side + '_images'):
            sample_shape.append(np.floor(np.array(image.shape) / np.array(self.opt.sample_size)))
            samples_per_image.append(sample_shape[-1].prod())
        setattr(self, 'samples_per_image_' + side, np.cumsum(samples_per_image))

    def init_samples_with_mask(self, side: Literal['A'] | Literal['B']):
        setattr(self, 'sample_shape_' + side, [])
        setattr(self, 'samples_to_skip_' + side, SortedDict())
        sample_shape = getattr(self, 'sample_shape_' + side)
        samples_to_skip = getattr(self, 'samples_to_skip_' + side)

        image = getattr(self, side + '_images')[0]
        mask = getattr(self, side + '_mask')

        sample_shape.append(np.floor(np.array(image.shape) / np.array(self.opt.sample_size)))
        max_sample_count = int(sample_shape[0].prod())
        skips = 0
        for i in range(max_sample_count):
            i_slice = self.get_nth_sample(sample_shape[0], i)
            if not mask[i_slice].all():
                skips += 1
            elif skips != 0:
                skips += 0 if len(samples_to_skip) == 0 else samples_to_skip.peekitem()[1]
                samples_to_skip[i - skips] = skips
                skips = 0
        if skips != 0:
            skips += 0 if len(samples_to_skip) == 0 else samples_to_skip.peekitem()[1]
            samples_to_skip[max_sample_count - skips] = skips

        unusable_samples = 0 if len(samples_to_skip) == 0 else samples_to_skip.peekitem()[1]
        setattr(self, 'samples_per_image_' + side, np.array([sample_shape[-1].prod() - unusable_samples]))

    def get_nth_sample(self, shape, n):
        z = int(n / (shape[1] * shape[2]))
        n -= z * (shape[1] * shape[2])
        y = int(n / shape[2])
        x = int(n - y * shape[2])

        z *= self.opt.sample_size[0]
        y *= self.opt.sample_size[1]
        x *= self.opt.sample_size[2]

        return (slice(z, z+self.opt.sample_size[0]), slice(y, y+self.opt.sample_size[1]), slice(x, x+self.opt.sample_size[2]))

    def samples_to_skip(self, side: Literal['A'] | Literal['B'], index):
        if not hasattr(self, 'samples_to_skip_' + side):
            return 0
        samples_to_skip = getattr(self, 'samples_to_skip_' + side)
        if len(samples_to_skip) == 0:
            return 0
        i = samples_to_skip.bisect_left(index)

        # The index is larger than the largest stored index
        if i >= len(samples_to_skip):
            return samples_to_skip.peekitem()[1]
        # The index references a key that's larger than index
        elif index < samples_to_skip.peekitem(i)[0]:
            i -= 1
          
        if i == -1:
            return 0
        else:
            return samples_to_skip.peekitem(i)[1]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helper functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        assert index < self.len
        indexA = index % self.samples_per_image_A[-1]
        indexB = index % self.samples_per_image_B[-1]
        A_image_index = np.argmax(self.samples_per_image_A > indexA)
        B_image_index = np.argmax(self.samples_per_image_B > indexB)

        A_index_in_image = indexA - (self.samples_per_image_A[A_image_index-1] if A_image_index > 0 else 0) + self.samples_to_skip('A', indexA)
        B_index_in_image = indexB - (self.samples_per_image_B[B_image_index-1] if B_image_index > 0 else 0) + self.samples_to_skip('B', indexB)
        A_shape = self.sample_shape_A[A_image_index]
        B_shape = self.sample_shape_B[B_image_index]

        A_img = self.A_images[A_image_index][self.get_nth_sample(A_shape, A_index_in_image)]
        B_img = self.B_images[B_image_index][self.get_nth_sample(B_shape, B_index_in_image)]
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_image': A_image_index,  'B_image': B_image_index}

    def __len__(self):
        """Return the total number of images."""

        return self.len
