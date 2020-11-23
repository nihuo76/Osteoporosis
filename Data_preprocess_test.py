# let's ignore the network we changed and focus on the data initializaion
# this code is devoted to debug the data initializaiton
from data_init import DPR_dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

# # load a model pre-trained pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#
# # replace the classifier with a new one, that has
# # num_classes which is user-defined
# num_classes = 2  # 1 class (person) + background
# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import torchvision.transforms as T
def get_transform():
    transforms = []
    transforms.append(T.Resize((1280, 2440)))
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = DPR_dataset(get_transform())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
new_target = []
for i in range(len(images)):
    new_target.append({})
for t in targets:
    for j, each_target in enumerate(targets[t]):
        new_target[j][t] = each_target


output = model(images, new_target)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions

