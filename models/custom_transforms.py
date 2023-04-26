from abc import ABC, abstractmethod
import torch
from torchvision.transforms import functional
import random

class AbstractTransform(ABC):
    def __init__(self, axes: list[str]=['xy']):
        assert all(map(lambda ax: len(ax) == 2 and ax in 'xyz', axes))
        self.axes = axes

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor):
        for ax in self.axes:
            if 'z' in ax:
                x = torch.swapaxes(x, -3, -1 if 'y' in ax else -2)
            x = self.transform(x)
            if 'z' in ax:
                x = torch.swapaxes(x, -1 if 'y' in ax else -2, -3)
        return x

class RandomDiscreteRotation(AbstractTransform):
    def __init__(self, angles: list[int], axes=['xy']):
        super().__init__(axes)
        self.angles = angles

    def transform(self, x):
        angle = random.choice(self.angles)
        return functional.rotate(x, angle, expand=True)

class RandomFlip(AbstractTransform):
    def __init__(self, axes=['xy']):
        super().__init__(axes)

    def transform(self, x):
        if random.getrandbits(1):
            x = functional.hflip(x)
        if random.getrandbits(1):
            x = functional.vflip(x)
        return x

class RandomPixelModifier:
    def __init__(self, change_probability, value):
        self.change_probability = change_probability
        self.value = value

    def __call__(self, x):
        mask = torch.rand(x.shape) < self.change_probability
        x = x.detach().clone()
        x[mask] = self.value
        return x

class RandomGaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = torch.normal(self.mean, self.std, x.shape)
        origin_dtype = torch.iinfo(x.dtype)
        return torch.clamp(x + noise, origin_dtype.min, origin_dtype.max).type(x.dtype)