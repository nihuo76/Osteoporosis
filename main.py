import torch
import torchvision
from data_init import DPR_dataset
import torch.utils.data as data
from DPR_utils.DPR_RCNN import FasterRCNN
from COCO_tools.engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import numpy as np

# import torchvision.transforms as T
# def get_transform():
#     transforms = []
#     transforms.append(T.Resize((1280, 2440)))
#     transforms.append(T.ToTensor())
#     # if train:
#     #     transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

device = torch.device('cuda')
dataset = DPR_dataset()
dataset_test = DPR_dataset()

indices = torch.randperm(len(dataset)).tolist()
dataset = data.Subset(dataset, indices[:-50])
dataset_test = data.Subset(dataset_test, indices[-50:])

data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
data_loader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=True)

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
model = FasterRCNN(backbone, num_classes=2)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params)
num_epochs = 3
epoch_loss_list = []

for epoch in range(num_epochs):
    _, epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    epoch_loss_list.append(epoch_loss)
    optimizer.step()
    # evaluate on the test dataset

evaluate(model, data_loader_test, device=device)

plt.figure()
plt.plot(np.arange(num_epochs), np.array(epoch_loss_list))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(fname='loss')
plt.close()

print("Finish")

