import torchvision
import torch
from DPR_utils.DPR_RCNN import FasterRCNN
from data_init import DPR_dataset

# there are 3 parts that we need to modify
# 1. FasterRCNN takes the argument box_predictor
#    the default box_predictor is FastRCNNPredictor which has the classification score
# 2. the box_predictor is an argument for initialize RoIHeads class
#    in RoIHeads class the box_predictor returns the class_logits. and class_logits
#    is an argument for fastrcnn_loss in training and postprocess_detections in testing
#    select_training_samples in RoIHeads must be adapted too
# 3. fastrcnn_loss must be changed. In such cases, we will have only 2 classes: foregraound and background
# 4. in postprocess_detections, class_logits is used to perform NMS on bounding box and retun
#    softmax(class_logits) as scores
# 5. RoIHeads attach the scores in the results that is going to be returned
# 6. FasterRCNN is a sub-class of GeneralizedRCNN and use it in initialization with the RoIHeads
# 7. the results returned by roi_heads in GeneralizedRCNN is used as "detections"
# 8. "detections" is then passed onto transform.postprocess for bounding box resize

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                 output_size=7,
#                                                 sampling_ratio=2)
model = FasterRCNN(backbone, num_classes=2)

import torchvision.transforms as T
def get_transform():
    transforms = []
    transforms.append(T.Resize((1280, 2440)))
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = DPR_dataset(get_transform())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# For inference
images, targets = next(iter(data_loader))
images = list(image for image in images)
new_target = []
for i in range(len(images)):
    new_target.append({})
for t in targets:
    for j, each_target in enumerate(targets[t]):
        new_target[j][t] = each_target

# targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, new_target)   # Returns losses and detections

print(output)







