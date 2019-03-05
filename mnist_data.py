import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms

import capsule as cps

_default_num_workers = 0
"""How many subprocesses to use for data loading."""


class MnistData:

    def __init__(self, root_folder):
        """
        Args:
            root_folder (string): Root folder where MNIST dataset is stored or will be downloaded.
        """
        self._root_folder = root_folder

    def download(self):
        """Downloads MNIST dataset into the root folder."""
        torchvision.datasets.MNIST(self._root_folder, download=True)

    def data_loader(self, batch_size, train, shuffle, transform=None, num_workers=_default_num_workers, digits=None):
        """
        Creates a data loader.

        Args:
            batch_size (int): How many samples per batch to load. ``-1`` means that all samples will be loaded in a
                single batch.
            train (bool): If ``True``, creates training data loader, otherwise test data loader.
            shuffle (bool): Whether to reshuffle the samples at every epoch.
            transform (callable, optional): A function that takes a PIL image and returns a transformed version.
            num_workers (int, optional): How many subprocesses to use for data loading. ``0`` means that the data will
                be loaded in the main process (default: ``0``).
            digits (sequence of ints, optional): What digits to load. ``None`` means that all digits will be loaded
                (default: ``None``).

        Returns:
            New data loader.
        """
        data_set = torchvision.datasets.MNIST(root=self._root_folder, train=train, transform=transform)

        if digits:
            mask = None

            for digit in digits:
                digit_mask = (data_set.train_labels if train else data_set.test_labels) == digit

                if mask is None:
                    mask = digit_mask
                else:
                    mask |= digit_mask

            if train:
                data_set.train_data = data_set.train_data[mask]
                data_set.train_labels = data_set.train_labels[mask]
            else:
                data_set.test_data = data_set.test_data[mask]
                data_set.test_labels = data_set.test_labels[mask]

        if batch_size == -1:
            batch_size = len(data_set.test_data)

        return torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=True, drop_last=True)

    def plain_loader(self, batch_size, train, shuffle=False, num_workers=_default_num_workers, digits=None):
        """
        Creates a data loader which provides original full-size samples.

        Args:
            batch_size (int): How many samples per batch to load. ``-1`` means that all samples will be loaded in a
                single batch.
            train (bool): If ``True``, creates training data loader, otherwise test data loader.
            shuffle (bool): Whether to reshuffle the samples at every epoch (default: ``False``).
            num_workers (int, optional): How many subprocesses to use for data loading. ``0`` means that the data will
                be loaded in the main process (default: ``0``).
            digits (sequence of ints, optional): What digits to load. ``None`` means that all digits will be loaded
                (default: ``None``).

        Returns:
            New data loader.
        """
        return self.data_loader(batch_size, train, shuffle, transform=PlainTransform(), num_workers=num_workers,
                                digits=digits)

    def rand_cut_loader(self, sample_grid_size, batch_size, train, shuffle=True, previous_layers=None,
                        num_workers=_default_num_workers, digits=None):
        """
        Creates a data loader which provides samples constructed from randomly translated digits.

        Args:
            sample_grid_size (sequence of ints) : Height and width of each sample.
            batch_size (int): How many samples per batch to load. ``-1`` means that all samples will be loaded in a
                single batch.
            train (bool): If ``True``, creates training data loader, otherwise test data loader.
            shuffle (bool): Whether to reshuffle the samples at every epoch (default: ``True``).
            previous_layers: Sequence of layers the input image is processed by to create a sample.
            num_workers (int, optional): How many subprocesses to use for data loading. ``0`` means that the data will
                be loaded in the main process (default: ``0``).
            digits (sequence of ints, optional): What digits to load. ``None`` means that all digits will be loaded
                (default: ``None``).

        Returns:
            New data loader.
        """
        assert len(sample_grid_size) == 2

        img_size = cps.input_grid_size_thru(previous_layers, sample_grid_size)
        transform = RandCut(img_size)

        if previous_layers:
            transform = transforms.Compose([transform, ThruLayers(previous_layers)])

        return self.data_loader(batch_size, train, shuffle, transform=transform, num_workers=num_workers, digits=digits)


class PlainTransform:
    """Transforms PIL image to tensor of shape (PIL image height, PIL image width, 1)."""

    def __call__(self, pic):
        return TF.to_tensor(pic).squeeze()[:, :, None]


class RandCut:
    """Transforms PIL image to tensor of shape (img_size[0], img_size[1], 1)."""

    def __init__(self, img_size):
        self._img_size = np.array(img_size)

    def __call__(self, pic):
        img = TF.to_tensor(pic).squeeze()
        img = RandCut._remove_black_border(img)
        return self._rand_crop(img)[:, :, None]

    @staticmethod
    def _remove_black_border(img):
        xn = img.sum(dim=0).nonzero()  # x indexes of nonzero columns.
        yn = img.sum(dim=1).nonzero()  # y indexes of nonzero rows.

        return img[yn[0]:(yn[-1] + 1), xn[0]:(xn[-1] + 1)]

    def _rand_crop(self, img):
        dy, dx = img.shape - self._img_size  # How much space there is for translation.
        # Negative dy or dx values means that the original image is smaller then the desired size, so there is no space
        # for translation and we need to add margin of size -dy or -dx.

        ry = np.random.randint(dy + 1) if dy >= 0 else np.random.randint(-dy + 1)
        rx = np.random.randint(dx + 1) if dx >= 0 else np.random.randint(-dx + 1)

        size_y, size_x = self._img_size

        if dy >= 0 and dx >= 0:
            cropped = img[ry:(ry + size_y), rx:(rx + size_x)]

        else:
            cropped = torch.zeros(size_y, size_x)

            if dy < 0 and dx < 0:
                cropped[ry:(ry + img.shape[0]), rx:(rx + img.shape[1])] = img
            elif dy < 0:  # and dx >= 0
                cropped[ry:(ry + img.shape[0]), :] = img[:, rx:(rx + size_x)]
            else:  # dy >= 0 and dx < 0
                cropped[:, rx:(rx + img.shape[1])] = img[ry:(ry + size_y), :]

        return cropped


class ThruLayers:
    """Processes samples through a sequence of layers."""

    def __init__(self, layers):
        self._layers = list(layers)

    def __call__(self, sample):
        sample = sample[None]

        with torch.no_grad():
            for layer in self._layers:
                sample = layer(sample)

        assert len(sample) == 1

        return sample.squeeze(dim=0)
