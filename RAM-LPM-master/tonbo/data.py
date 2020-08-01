from pathlib import Path

import torch
from torchvision import datasets, transforms
import numpy as np
from scipy.ndimage.interpolation import geometric_transform
from scipy import ndimage
from torch.utils.data import Dataset


class ExpandDataset(Dataset):
    def __init__(
        self, dataset, width: int = 60, height: int = 60, background: float = 0.0,
    ):
        self.dataset = dataset
        self.width = width
        self.height = height
        self.background = background
        self.size = len(dataset)
        data, y = self.dataset[0]
        C, W, H = data.shape
        self._x_list = [np.random.randint(self.width - W) for _ in range(self.size)]
        self._y_list = [np.random.randint(self.height - H) for _ in range(self.size)]

    def __getitem__(self, i):
        data, y = self.dataset[i]
        C, W, H = data.shape
        assert W <= self.width, "Width must be larger than width of data image"
        assert H <= self.height, "Height must be larger than height of data image"
        _canvas = torch.zeros(C, self.width, self.height)
        _canvas.fill_(self.background)
        _x = self._x_list[i]
        _y = self._y_list[i]
        _canvas[:, _x : _x + W, _y : _y + H] = data
        return _canvas, y

    def __len__(self):
        return self.size


class RotatePrepocessDataset(Dataset):
    def __init__(
        self,
        dataset,
        rootpath: str = "./dataset",
        rotation_angle: float = 0.0,
        clockwise: bool = True,
        polar_coordinate: bool = False,
        bg_value: float = 0.5,
    ):
        self.dataset = dataset
        self.rootpath = rootpath
        self.rotation_angle = rotation_angle
        self.polar_coordinate = polar_coordinate
        self.clockwise = clockwise
        self.size = len(dataset)
        self.index = 0
        self.bg_value = bg_value

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        if self.index >= self.size:
            raise StopIteration
        _index = self.index
        self.index += 1
        return self[_index]

    def __getitem__(self, i):
        data, y = self.dataset[i]
        if self.rotation_angle != 0.0:
            data = self._rotate(data)
        if self.polar_coordinate:
            data, (rads, angs) = self._transform2polar(data)
        return data, y

    def _rotate(self, img):
        _img = img[0]
        bg_value = 0  # this is regarded as background's value black
        _img = ndimage.rotate(
            _img, self.rotation_angle, reshape=False, cval=self.bg_value
        )
        img[0, :, :] = torch.FloatTensor(_img)
        return img

    def _transform2polar(self, img, order=1):
        """
        Transform img to its polar coordinate representation.

        order: int, default 1
        Specify the spline interpolation order.
        High orders may be slow for large images.
        """
        # max_radius is the length of the diagonal
        # from a corner to the mid-point of img.
        _img = img[0]
        max_radius = 0.5 * np.linalg.norm(_img.shape)

        def transform(coords):
            # Put coord[1] in the interval, [-pi, pi]
            theta = 2 * np.pi * coords[1] / (_img.shape[1] - 1.0)

            radius = max_radius * coords[0] / _img.shape[0]

            i = 0.5 * _img.shape[0] - radius * np.sin(theta)
            j = radius * np.cos(theta) + 0.5 * _img.shape[1]
            return i, j

        _img = _img.data.numpy()
        polar = geometric_transform(_img, transform, order=order)
        rads = max_radius * np.linspace(0, 1, _img.shape[0])
        angs = np.linspace(0, 2 * np.pi, _img.shape[1])
        img[0, :, :] = torch.FloatTensor(polar)

        return img, (rads, angs)

    def __len__(self):
        return self.size


if __name__ == "__main__":
    from torchvision import datasets, transforms

    data_root = "~/.mnist"

    train_dataset = datasets.MNIST(
        data_root,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),]),
    )
    train_dataset = ExpandDataset(train_dataset, 100, 100, 0)
    print(train_dataset[0])
    print(train_dataset[0])
