
import torch
import torch.nn as nn
from torchvision import models


def load_model(model_pth, num_classes=196):
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))

    model = models.efficientnet_v2_s().to('cpu')
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
