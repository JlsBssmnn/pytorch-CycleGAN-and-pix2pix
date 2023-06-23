import sys
from pathlib import Path
from models.custom_transforms import Scaler, create_transform
import skimage
import numpy as np
import torch

class AffinityToSeg:
    def __init__(self, opt):
        path = str(Path(__file__).parent.parent.parent)
        if path not in sys.path:
            sys.path.append(path)

        from evaluation.evaluate_brainbows import run_mws, get_foreground_mask
        self.run_mws = run_mws
        self.get_foreground_mask = get_foreground_mask

        self.scaler = Scaler(-1, 1, 0, 1)
        self.opt = opt
        self.offsets = opt.evaluation_config.offsets

    def transform(self, image, label):
        """
        Transform an image in affinity representation to a segmentation.
        """
        if label.startswith('real'):
            return label, None

        image = self.scaler(image).detach().cpu().numpy()
        if image.ndim == 5:
            image = image[0]

        if label.startswith('original'):
            return 'real_'  + label[-1], (image * 255).astype(np.uint8)

        foreground_mask = self.get_foreground_mask(image[1:], image[0], self.offsets, 0.5)
        seg = self.run_mws(image[1:], self.offsets,
                      stride=(1, 1, 1),
                      foreground_mask=foreground_mask,
                      seperating_channel=self.opt.evaluation_config.seperating_channel,
                      invert_dam_channels=True,
                      randomize_bounds=False)

        seg = skimage.measure.label(seg)
        if seg.max() <= 65_535:
            seg = seg.astype(np.uint16)
        elif seg.max() <= 4_294_967_295:
            seg = seg.astype(np.uint32)

        return label, seg

class ToUint8:
    def __init__(self, opt):
        self.tanh_to_uint8 = create_transform('tanh_to_uint8')

    def transform(self, image, label):
        image = self.tanh_to_uint8(image).type(torch.uint8)
        image = image.detach().cpu().numpy()
        if image.ndim == 5:
            image = image[0]
        return label, image

