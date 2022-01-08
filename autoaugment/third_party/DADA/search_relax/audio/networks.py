import torch

from autoaugment.domain.audio.classification.models.densenet import DenseNet
from autoaugment.domain.audio.classification.models.resnet import ResNet
from autoaugment.domain.audio.classification.models.inception import Inception


def get_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['model'] == "densenet":
        model = DenseNet(config['dataset_name'],
                         config['pretrained']).to(device)
    elif config['model'] == "resnet":
        model = ResNet(config['dataset_name'], config['pretrained']).to(device)
    elif config['model'] == "inception":
        model = Inception(config['dataset_name'],
                          config['pretrained']).to(device)
    return model
