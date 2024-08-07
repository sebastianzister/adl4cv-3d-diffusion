from torchvision import datasets, transforms
from base import BaseDataLoader
#from dataset import PointDetailDataset, PUNET_Dataset_1, ShapeNet15kPointCloudsAugmented, PVDDataset
from dataset import PointDetailDataset, ShapeNet15kPointCloudsAugmented, PVDDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class PointDetailDataLoader(BaseDataLoader):
    """
    PointDetail data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = PointDetailDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#class PUNETDataLoader(BaseDataLoader):
#    """
#    PUNET data loading using BaseDataLoader
#    """
#    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#        self.dataset = PUNET_Dataset_1()
#        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ShapeNetAugmentedDataLoader(BaseDataLoader):
    """
    ShapeNetAugmented data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, training=True, size=None, pin_memory=False, tr_sample_size=4096, down_ratio=2, random_downsample=False, noise=False, **kwargs):
        self.dataset = ShapeNet15kPointCloudsAugmented(down_ratio=down_ratio, tr_sample_size=tr_sample_size, length=size, random_downsample=random_downsample, noise=noise, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory=pin_memory)

class PVDDataLoader(BaseDataLoader):
    """
    PVD data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=0, training=True, size=None, pin_memory=False, categories=["car"]):
        print("PVDDataLoader BS: ", batch_size)
        print("PVDDataLoader categories: ", categories)
        self.dataset = PVDDataset(categories=categories)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory=pin_memory)