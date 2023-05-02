from abc import ABC, abstractmethod
import torch
import torchvision
from torchvision.transforms import functional
import random

class Transform:
    """
    This class represents a transform. It can apply certain
    pre- and post-processing steps that are required for a
    transformation.
    """
    def __init__(self, name, slice_string=None, **kwargs):
        self.wrapper = []
        if slice_string is not None:
            self.wrapper.append(Slicer(slice_string))

        self.transform = globals()[name](**kwargs)

    def transform_image(self, x):
        inputs = []
        x = x.detach().clone()
        for t in self.wrapper:
            inputs.append(x)
            x = t.pre(x)
        x = self.transform(x)
        for t, out in zip(reversed(self.wrapper), reversed(inputs)):
            x = t.post(x, out)
        return x

    def __call__(self, x):
        if x.ndim == 4:
            return self.transform_image(x)
        elif x.ndim == 5:
            output = torch.empty(x.shape)
            for i in range(x.shape[0]):
                output[i] = self.transform_image(x[i])
            return output
        else:
            raise NotImplementedError('Transform cannot be applied to dimensionality %d' % x.ndim)


class Slicer:
    """
    A processing step for a transformation. It will only apply
    the transformation on a slice of the input.
    """
    def __init__(self, slice_string=None):
        if slice_string is None:
            self.slice = None
            return

        self.slice = slice_string.split(',')
        self.slice = [x.split(':') for x in self.slice]
        self.slice = [[eval(y) for y in x] for x in self.slice]
        self.slice = tuple([slice(*x) for x in self.slice])

    def pre(self, x):
        return x[self.slice]

    def post(self, x, original):
        original[self.slice] = x
        return original

class RandomDiscreteRotation:
    """
    Randomly rotates the input by one of the given angles.
    """
    def __init__(self, angles: list[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return functional.rotate(x, angle, expand=True)

class RandomFlip:
    """
    Randomly flips the input vertically and/or horizontally.
    """
    def __call__(self, x):
        if random.getrandbits(1):
            x = functional.hflip(x)
        if random.getrandbits(1):
            x = functional.vflip(x)
        return x

class RandomPixelModifier:
    """
    Creates a random mask where each pixel is included with a
    probability of `change_probability`. All voxels of that
    mask are then set to the given value.
    """
    def __init__(self, change_probability, value):
        self.change_probability = change_probability
        self.value = value

    def __call__(self, x):
        mask = torch.rand(x.shape) < self.change_probability
        x = x.detach().clone()
        x[mask] = self.value
        return x

class RandomGaussianNoise:
    """
    Applies a gaussian noise to the input.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = torch.normal(self.mean, self.std, x.shape)
        if x.dtype.is_floating_point:
            origin_dtype = torch.finfo(x.dtype)
        else:
            origin_dtype = torch.iinfo(x.dtype)
        return torch.clamp(x + noise, origin_dtype.min, origin_dtype.max).type(x.dtype)

class Threshold():
    """
    Thresholds the input: All values below the threshold are
    mapped to the lower value, all other values are mapped
    to the upper value.
    """
    def __init__(self, lower, upper, threshold=None):
        assert lower < upper

        self.lower = lower
        self.upper = upper

        if threshold is None:
            self.threshold = (lower + upper) / 2
        else:
            self.threshold = threshold

    def __call__(self, x):
        x[x >= self.threshold] = self.upper
        x[x < self.threshold] = self.lower
        return x

class SetValue:
    """
    Sets the value of every voxel to the specified value.
    """
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        x[:] = self.value
        return x


class Scaler():
    """
    Scales the values of the input. The input value range is between
    `in_min` and `in_max`. The values are scaled to the range between
    `out_min` and `out_max`.
    """
    def __init__(self, in_min, in_max, out_min, out_max):
        assert in_min < in_max and out_min < out_max

        in_diff = abs(in_max - in_min)
        out_diff = abs(out_max - out_min)
        self.in_min = in_min
        self.in_max = in_max
        self.scaling = out_diff / in_diff
        self.offset = out_min - in_min * self.scaling

    def __call__(self, x):
        x = torch.clamp(x, self.in_min, self.in_max)
        return x * self.scaling + self.offset

class ChannelModifier:
    """
    Manipulates the number of channels in an image.
    """

    def __init__(self, x, target_channel_count=None, mode='value', channels_to_delete=[]):
        """
        Note: Either target_channel_count or channels_to_delete must be provided.

        Parameters:
        -------
        target_channel_count: How many channels the output of this transform should have
        mode: How to interpret the `x` parameter to get the values of the new channels
            ('value': x is used directly as the value,
             'index': x is used as index into the channels. The channel at that index is copied and used for the new channels)
        channels_to_delete: Which channels should be deleted.
        """
        assert target_channel_count is not None or channels_to_delete

        self.target_channel_count = target_channel_count
        self.mode = mode
        self.channels_to_delete = channels_to_delete
        self.x = x


    def __call__(self, x):
        assert x.ndim == 4

        if self.channels_to_delete:
            to_keep = list(set(range(x.shape[0])) - set(self.channels_to_delete))
            x = x[to_keep]

        if self.target_channel_count is None or x.shape[0] == self.target_channel_count:
            return x

        if x.shape[0] > self.target_channel_count:
            x = x[:self.target_channel_count]
        else:
            missing = self.target_channel_count - x.shape[0]
            if self.mode == 'value':
                y = torch.full((missing,) + x.shape[1:], self.x)
                x = torch.cat((x, y), dim=0)
            elif self.mode == 'index':
                y = x[self.x].detach().clone()
                y = torch.stack((y,)*missing, dim=0)
                x = torch.cat((x, y), dim=0)
            else:
                raise NotImplementedError('Mode [%s] is not implemented' % self.mode)

        return x


def get_transform_by_string(transform_string: str):
    if transform_string == 'tanh_to_uint8':
        return Scaler(-1, 1, 0, 255)
    elif transform_string == 'sigmoid_to_uint8':
        return Scaler(0, 1, 0, 255)
    else:
        raise NotImplementedError('Transform [%s] is not found' % transform_string)

def create_transform(transform_list: list | str | None):
    """
    Creates a transformation out of a list of transformations. Each transformation can be specified
    in 2 ways: as a dict, that must have a `name` key, that specifies which transformation shall be applied,
    the other key value pairs are passed to the transformation as parameters. Or a string that represents
    a hardcoded transformation.
    """
    transformations = []
    if transform_list is None:
        return torchvision.transforms.Compose(transformations)
    elif type(transform_list) == str:
        return get_transform_by_string(transform_list)

    for t in transform_list:
        if type(t) == str:
            transformations.append(get_transform_by_string(t))
        else:
            assert 'name' in t, "A transformation must have a name"
            transformations.append(Transform(**t))
    return torchvision.transforms.Compose(transformations)
