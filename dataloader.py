import os
import numpy as np
import csv
import torch
from PIL import Image


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
        
        img = Image.open(img_path).convert("RGB")
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)