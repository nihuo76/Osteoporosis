from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from flipping import flipping
from getimagexywh import getimagexywh
import torch
import torchvision.transforms as T

class DPR_dataset(Dataset):
    def __init__(self):
        transforms = []
        transforms.append(T.Resize((1280, 2440)))
        transforms.append(T.ToTensor())
        self.transforms = T.Compose(transforms)
        Osteo_dataset_folder = os.path.join(os.getcwd(), "Osteoporosis_Dataset")
        osteo_csv = os.path.join(Osteo_dataset_folder, "data_info_v1.2.csv")
        osteo_csv = pd.read_csv(osteo_csv)
        self.osteo_csv = osteo_csv

    def __len__(self):
        # return len(self.osteo_csv)*2
        return len(self.osteo_csv)*2

    def __getitem__(self, idx):
        # note that the person is idx // len(self.osteo_csv)
        # while the degree is 90*(idx % len(self.osteo_csv))
        person_idx = idx % len(self.osteo_csv)
        flipping_flag = (idx >= len(self.osteo_csv))
        personname = self.osteo_csv["Filename"][person_idx]
        BMD_score = self.osteo_csv["T_score"][person_idx]
        # let's ignore the data augmentation during developement period
        if flipping_flag:
            PIL_image, xy = flipping(personname)
        else:
            PIL_image, xy = getimagexywh(personname)
        # PIL_image, xy = getimagexywh(personname)
        PIL_image = PIL_image.convert("RGB")
        PIL_image = self.transforms(PIL_image)
        xy = torch.as_tensor(xy)
        boxes = torch.zeros((8, 4), dtype=PIL_image.dtype)
        boxes[:, :2] = xy
        boxes[:, 2:] = xy + 50
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        labels = torch.ones((8,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        BMD_score = torch.tensor([BMD_score], dtype=torch.float64)
        iscrowd = torch.zeros((8,), dtype=torch.int64)
        target["boxes"] = boxes
        target["BMD"] = BMD_score
        target["area"] = area
        target["image_id"] = image_id
        target["labels"] = labels
        target["iscrowd"] = iscrowd

        return PIL_image, target
