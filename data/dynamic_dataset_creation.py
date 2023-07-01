"""
Utilities that can create datasets at runtime.
"""

import numpy as np
import itertools
from typing import Literal


def create_offsets(radius: float, neighborhood: Literal[6, 18, 26], element_size_um=(0.2, 0.1, 0.1), distance=None):
    """
    Creates offset values for an affinity representation. The repulsive offsets are based on a cube with
    the given radius. The neighborhood parameter specifies how many neighbors of a voxel should be considered
    for the offsets. The element_size_um is needed to scale the distances in isotropic images.
    """
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    element_size_um = np.array(element_size_um)
    element_size_um = 1 / element_size_um
    element_size_um /= element_size_um.max()

    if neighborhood == 6:
        level = 1
    elif neighborhood == 18:
        level = 2
    elif neighborhood == 26:
        level = 3
    else:
        raise NotImplementedError(f"Neighborhood of {neighborhood} not implemented")

    for i in range(level):
        for combination in itertools.combinations([0, 1, 2], i + 1):
            if distance is None:
                offset = np.array([0, 0, 0])
                for j in combination:
                    offset[j] = int(radius * element_size_um[j])
            elif distance == 'euclidean':
                offset = np.array([0, 0, 0], dtype=float)
                for j in combination:
                    offset[j] = radius
                offset *= radius / np.linalg.norm(offset)
                offset *= element_size_um
                offset = np.round(offset).astype(int)
            else:
                raise NotImplementedError(f"Distance {distance} is not implemented")

            offsets.append(tuple(offset))

    return offsets

def create_brainbow_affinities(image, offsets, bg_measure, bg_threshold, dist_measure='norm', norm_order=2):
    """
    Creates an affinity image of a colored brainbow image.

    bg_measure: A numpy function that is used to aggregate color values to one number (e.g. sum or mean).
    dist_measure: Which mesaure to use for converting the distance vector between two colors to a value.
    bg_threshold: Every value lower than this threshold is treated as background.
    norm_order: Which norm to use for computing the affinities.
    """
    offsets = np.array(offsets)
    affinities = np.empty((len(offsets) + 1, *image.shape[-3:]), dtype=np.float32)
    c = image.shape[0]

    affinities[0] = eval(f"np.{bg_measure}(image, axis=0)")
    if bg_threshold is not None:
        affinities[0] = affinities[0] > bg_threshold

    # scale image to values between 0 and 1
    if image.min() < 0 or image.max() > 1:
        scaling = (1 / (image.max() - image.min()))
        image = image * scaling - image.min() * scaling  

    for i, offset in enumerate(offsets):
        rolled_image = np.roll(image, -offset, (-3, -2, -1))
        diff = image - rolled_image

        if dist_measure == 'norm':
            values = np.linalg.norm(diff, ord=norm_order, axis=0)

            if norm_order == 0:
                values /= c
            else:
                values /= c ** (1.0 / norm_order)
            values = 1 - values
        elif dist_measure == 'mse':
            values = 1 - np.mean(diff ** 2, axis=0)
        elif dist_measure == 'mae':
            values = 1 - np.mean(np.abs(diff), axis=0)
        else:
            raise NotImplementedError(f"The dist_measure {dist_measure} is not implemented")

        affinities[i + 1] = values
    return affinities

def synth_brainbow_to_affinities(image, offsets):
    """
    This function converts a colored brainbow image that has been synthetically generated to the affinity
    representation. Note that the special property of these images is that the color of one neuron is always constant,
    meaning the color value doesn't change across the neuron at all. Also the background are only voxel with value zero.
    """
    return create_brainbow_affinities(image, offsets, 'min', 1e-10, 'norm', 0)
