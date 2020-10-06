import torch
import torchvision
from torchvision import models, transforms
from .OQA_model import OQA_model


def OQA_loader(model_path):
    model = OQA_model(n_class=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #create transformer
    transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.2393,0.4637,0.4452),(0.1212,0.2222,0.0900))])

    return model, transformer
