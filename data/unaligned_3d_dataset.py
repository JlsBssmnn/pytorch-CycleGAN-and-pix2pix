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
import random
from typing import Literal

import numpy as np
import numpy.typing as npt
from sortedcontainers import SortedDict
from data.base_dataset import BaseDataset, get_transform_3d
from data.h5_folder import get_datasets
from util.logging_config import logging
from util.misc import StoreDictKeyPair


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
        parser.add_argument('--border_offset_A', type=int, nargs=3, default=[0, 0, 0], help='How much space around ' + \
                'the z-,y- and x-borders of the image are not used for sampling images from A')
        parser.add_argument('--border_offset_B', type=int, nargs=3, default=[0, 0, 0], help='How much space around ' + \
                'the z-,y- and x-borders of the image are not used for sampling images from B')
        parser.add_argument('--sample_size', type=int, nargs=3, required=True, help='The size along z-,y- and x-axis for a sampled image')
        parser.add_argument('--datasetA_file', type=str, required=True, help='The name of the file that contains dataset A')
        parser.add_argument('--datasetB_file', type=str, required=True, help='The name of the file that contains dataset B')
        parser.add_argument('--datasetA_names', type=str, nargs='+', help='The names of the datasets in the file that belong to dataset A')
        parser.add_argument('--datasetB_names', type=str, nargs='+', help='The names of the datasets in the file that belong to dataset B')
        parser.add_argument('--datasetA_mask', type=str, default=None, help='The name of the dataset to use as mask for dataset A')
        parser.add_argument('--datasetB_mask', type=str, default=None, help='The name of the dataset to use as mask for dataset B')
        parser.add_argument('--dataset_length', type=str, default='max', choices=['min', 'max'], help='How to determine the size of the entire dataset given the sizes of dataset A and dataset B')
        parser.add_argument('--no_normalization', action='store_true', help='If specified the samples are not normalized')
        parser.add_argument('--dataset_stride', type=int, nargs=3, default=None, help='The stride of the kernel that moves thorugh the data array to sample inputs')
        parser.add_argument('--dataset_transforms_A', action=StoreDictKeyPair, default=None, nargs="+",
                            metavar="KEY=VAL", help='Transformations that are applied to the sampled patches from dataset A.')
        parser.add_argument('--dataset_transforms_B', action=StoreDictKeyPair, default=None, nargs="+",
                            metavar="KEY=VAL", help='Transformations that are applied to the sampled patches from dataset B.')
        parser.add_argument('--datasetA_creation_func', type=object, default=None, help='A function that is applied to the loaded data of dataset A')
        parser.add_argument('--datasetB_creation_func', type=object, default=None, help='A function that is applied to the loaded data of dataset B')
        parser.add_argument('--datasetA_random_sampling', type=bool, default=False, help='If true samples for dataset A are randomly samples from the array')
        parser.add_argument('--datasetB_random_sampling', type=bool, default=False, help='If true samples for dataset B are randomly samples from the array')

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

        self.samples_per_image_A: npt.NDArray
        self.samples_per_image_B: npt.NDArray
        self.samples_per_image_A: npt.NDArray
        self.samples_per_image_B: npt.NDArray
        self.sample_count_per_axis_A = []
        self.sample_count_per_axis_B = []
        self.samples_to_skip_A = None
        self.samples_to_skip_B = None

        if opt.dataset_stride:
            assert all(map(lambda x: x > 0, opt.dataset_stride)), 'Invalid dataset_stride'
            self.dataset_stride = np.array(opt.dataset_stride, dtype=int)
        else:
            self.dataset_stride = np.array(opt.sample_size, dtype=int)

        self.A_images = get_datasets(os.path.join(opt.dataroot, opt.phase + 'A', opt.datasetA_file), opt.datasetA_names, [opt.datasetA_mask])
        self.B_images = get_datasets(os.path.join(opt.dataroot, opt.phase + 'B', opt.datasetB_file), opt.datasetB_names, [opt.datasetB_mask])
        assert all([x.ndim == 4 for x in self.A_images]) and all([x.ndim == 4 for x in self.B_images]), \
            'Images must be 4D (3 dimensions + color channels)'

        offset_slicing_A = tuple([slice(offset, -offset if offset > 0 else None) for offset in opt.border_offset_A])
        offset_slicing_B = tuple([slice(offset, -offset if offset > 0 else None) for offset in opt.border_offset_B])
        self.A_images = [arr[(slice(None),) + offset_slicing_A] for arr in self.A_images]
        self.B_images = [arr[(slice(None),) + offset_slicing_B] for arr in self.B_images]

        self.original_A_images = self.original_B_images = None

        if opt.datasetA_creation_func is not None:
            self.original_A_images = self.A_images
            self.A_images = [opt.datasetA_creation_func(x) for x in self.A_images]
        if opt.datasetB_creation_func is not None:
            self.original_B_images = self.B_images
            self.B_images = [opt.datasetB_creation_func(x) for x in self.B_images]

        if opt.datasetA_mask:
            assert len(self.A_images) == 1
            self.A_mask = get_datasets(os.path.join(opt.dataroot, opt.phase + 'A', opt.datasetA_file), [opt.datasetA_mask])[0][offset_slicing_A]
            assert self.A_mask.ndim == 3, 'A mask must be 3D'
            self.init_samples_with_mask('A')
        else:
            self.init_samples_no_mask('A')
            self.A_mask = None

        if opt.datasetB_mask:
            assert len(self.B_images) == 1
            self.B_mask = get_datasets(os.path.join(opt.dataroot, opt.phase + 'B', opt.datasetB_file), [opt.datasetB_mask])[0][offset_slicing_B]
            assert self.B_mask.ndim == 3, 'A mask must be 3D'
            self.init_samples_with_mask('B')
        else:
            self.init_samples_no_mask('B')
            self.B_mask = None
        
        assert self.samples_per_image_A[-1] > 0 and self.samples_per_image_B[-1] > 0, \
            "One of the datasets is empty (could be due to a mask)"

        if opt.datasetA_random_sampling and opt.datasetB_random_sampling:
            assert opt.max_dataset_size <= float('inf')
            self.len = opt.max_dataset_size
        elif opt.dataset_length == 'min':
            self.len = int(min(self.samples_per_image_A[-1], self.samples_per_image_B[-1]))
        elif opt.dataset_length == 'max':
            self.len = int(max(self.samples_per_image_A[-1], self.samples_per_image_B[-1]))
        else:
            logging.error('Invalid dataset_length parameter!')
            exit(1)

        self.transform_A = get_transform_3d(self.opt, 'A')
        self.transform_B = get_transform_3d(self.opt, 'B')

    def init_samples_no_mask(self, side: Literal['A'] | Literal['B']):
        samples_per_image = []
        sample_count_per_axis = getattr(self, 'sample_count_per_axis_' + side)

        for image in getattr(self, side + '_images'):
            sample_count_per_axis.append(np.floor(
                (np.array(image.shape[1:]) - self.opt.sample_size) / self.dataset_stride + 1
            ))
            samples_per_image.append(sample_count_per_axis[-1].prod())
        setattr(self, 'samples_per_image_' + side, np.cumsum(samples_per_image, dtype=int))

    def init_samples_with_mask(self, side: Literal['A'] | Literal['B']):
        setattr(self, 'samples_to_skip_' + side, SortedDict())
        sample_count_per_axis = getattr(self, 'sample_count_per_axis_' + side)
        samples_to_skip = getattr(self, 'samples_to_skip_' + side)

        image = getattr(self, side + '_images')[0]
        mask = getattr(self, side + '_mask')

        sample_count_per_axis.append(np.floor(
            (np.array(image.shape[1:]) - self.opt.sample_size) / self.dataset_stride + 1
        ))
        max_sample_count = int(sample_count_per_axis[0].prod())
        skips = 0
        for i in range(max_sample_count):
            i_slice = self.get_nth_sample(sample_count_per_axis[0], i)
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
        setattr(self, 'samples_per_image_' + side, np.array([max_sample_count - unusable_samples], dtype=int))

    def get_nth_sample(self, shape, n):
        z = int(n / (shape[1] * shape[2]))
        n -= z * (shape[1] * shape[2])
        y = int(n / shape[2])
        x = int(n - y * shape[2])

        z *= self.dataset_stride[0]
        y *= self.dataset_stride[1]
        x *= self.dataset_stride[2]

        return (slice(z, z+self.opt.sample_size[0]), slice(y, y+self.opt.sample_size[1]), slice(x, x+self.opt.sample_size[2]))

    def samples_to_skip(self, side: Literal['A'] | Literal['B'], index):
        samples_to_skip = getattr(self, 'samples_to_skip_' + side)
        if not samples_to_skip:
            return 0
        elif len(samples_to_skip) == 0:
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

    def get_item_from_dataset(self, index, dataset: Literal['A', 'B'], serial_batches):
        if getattr(self.opt, f'dataset{dataset}_random_sampling'):
            image_index = random.randint(0, len(getattr(self, f'samples_per_image_{dataset}')) - 1)
            image_shape = getattr(self, f'{dataset}_images')[image_index].shape
            slice_start = np.random.randint((0, 0, 0), np.array(image_shape[-3:]) - self.opt.sample_size)
            s = (slice(None),) + tuple([slice(slice_start[i], slice_start[i] + self.opt.sample_size[i]) for i in range(3)])

            mask = getattr(self, f'{dataset}_mask')
            if mask is not None:
                while not mask[s[1:]].all():
                    slice_start = np.random.randint((0, 0, 0), np.array(image_shape[-3:]) - self.opt.sample_size)
                    s = (slice(None),) + tuple([slice(slice_start[i], slice_start[i] + self.opt.sample_size[i]) for i in range(3)])
        else:
            dataset_index = index % getattr(self, f'samples_per_image_{dataset}')[-1]

            if serial_batches:
                image_index = np.argmax(getattr(self, f'samples_per_image_{dataset}') > dataset_index)
            else:
                image_index = random.randint(0, len(getattr(self, f'samples_per_image_{dataset}')) - 1)

            if serial_batches:
                index_in_image = dataset_index \
                    - (getattr(self, f'samples_per_image_{dataset}')[image_index-1] if image_index > 0 else 0) \
                    + self.samples_to_skip(dataset, dataset_index)
            else:
                index_in_image = random.randint(0, getattr(self, f'samples_per_image_{dataset}')[image_index] - \
                    (getattr(self, f'samples_per_image_{dataset}')[image_index - 1] if image_index > 0 else 0) - 1)

            shape = getattr(self, f'sample_count_per_axis_{dataset}')[image_index]

            s = (slice(None),) + self.get_nth_sample(shape, index_in_image)

        img = getattr(self, f'{dataset}_images')[image_index][s]
        img = getattr(self, f'transform_{dataset}')(img)

        item = {dataset: img, f'{dataset}_image': image_index}

        if getattr(self, f'original_{dataset}_images') is not None:
            item |= {f'original_{dataset}': getattr(self, f'transform_{dataset}')(getattr(self, f'original_{dataset}_images')[image_index][s])}

        return item

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names.
        """
        assert index < self.len

        item = self.get_item_from_dataset(index, 'A', True) | \
               self.get_item_from_dataset(index, 'B', self.opt.serial_batches)

        return item

    def __len__(self):
        """Return the total number of images."""

        return self.len
