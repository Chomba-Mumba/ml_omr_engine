import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#load pre-trained dataset
model = fasterrcnn_resnet50_fpn(pretrained=True)

#number of classes in dataset
num_classes =  3

#get number of input features for classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

#replace head of model with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

