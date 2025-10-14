from torchvision.datasets import VisionDataset
from PIL import Image


class DPPSet(VisionDataset):

    def __init__(self, images, labels, transform=None):
        # super().__init__(root, transform)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
