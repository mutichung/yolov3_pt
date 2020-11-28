import os
import numpy as np
import csv
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

class Rotate(object):
    """Rotate the image to landscape.
    Args:
        None
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        h, w = img.shape[1:]
        if h > w:
            img = torch.transpose(img, 1, 2)
            target['boxes'] = target['boxes'][:,[1,0,3,2]]
        return {'img': img, 'target': target}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image, target = sample['img'], sample['target']
        h, w = image.shape[1:]
        assert h <= w, 'image isn\'t in landscape.'
        assert w == 1024, 'image size is incorrect.'
        assert self.output_size[1] == 1024, 'output size is incorrect.'

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = TF.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]
        target['boxes'][:, ::2] = target['boxes'][:, ::2] * new_w / w
        target['boxes'][:, 1::2] = target['boxes'][:, 1::2] * new_h / h

        return {'img': img, 'target': target}

def get_transform(output_size=(768, 1024)):
    return torchvision.transforms.Compose([
        Rotate(),
        Rescale(output_size)
    ])


class OpenImagesDataset(object):
    def __init__(self, root, transforms=None, operation='test'):
        self.root = root
        self.operation = operation
        self.transforms = transforms
        self.img_dir_path = os.path.join(root, self.operation, 'footwear')
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(self.img_dir_path)))
        # remove the last item of the list, which is the label directory
        self.imgs.pop()
        # load all box text files and sort them to align with the image list
        self.boxes = list(sorted(os.listdir(os.path.join(self.img_dir_path, 'labels'))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.img_dir_path, self.imgs[idx])
        box_path = os.path.join(self.img_dir_path, 'labels', self.boxes[idx])
        
        img = Image.open(img_path)#.convert("RGB")
        img = TF.to_tensor(img)
        # img.unsqueeze_(0)

        boxes = []
        with open(box_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                x_tl = float(row[1])
                y_tl = float(row[2])
                x_br = float(row[3])
                y_br = float(row[4])
                boxes.append([x_tl, y_tl, x_br, y_br])
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float64)
        # the number of objects equals to the number of rows
        num_objs = boxes.shape[0]
        # all objects comes with label 1
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        sample = {'img': img, 'target': target}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.imgs)