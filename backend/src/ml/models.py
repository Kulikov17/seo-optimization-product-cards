
import torch
import torch.nn as nn
from torchvision import models


def load_model(model_pth, num_classes=196):
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))

    model = models.efficientnet_v2_s().to('cpu')
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def build_model(pretrained=True, fine_tune=True, num_classes=196):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    model = models.efficientnet_v2_s(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    return model
