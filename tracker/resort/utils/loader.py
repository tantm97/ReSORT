from __future__ import print_function

import torch
# from torch.nn import DataParallel

# from tracker.extractor.models import *
from tracker.extractor.models.resnet import resnet_face18, resnet_face34, resnet_face50

def load_model(model_path, backbone='resnet34', load_to_cpu=False, pretrained=False):
    if backbone == 'resnet18':
        model = resnet_face18(pretrained)
    elif backbone == 'resnet34':
        model = resnet_face34(pretrained)
    elif backbone == 'resnet50':
        model = resnet_face50(pretrained)

    # model = DataParallel(model)
    check_cuda_available = False
    if not load_to_cpu:
        if torch.cuda.is_available():
            print("Cuda is available.")
            check_cuda_available = True
        else:
            print("Cuda is not available.")

    if check_cuda_available:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))
        model.load_state_dict(pretrained_dict)
        model = model.to(device)
    else:
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained_dict)

    print("Model is on GPU: ", next(model.parameters()).is_cuda)
    
    return model
