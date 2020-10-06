from torchvision import models, transforms
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import tqdm
import os, sys
import torch.nn as nn
from PIL import Image
import glob
import OQA_model

class CAM_Heatmap_generator():
    def __init__(self, model=None, data_transformer=None, finalconv_name=None, FCL_index=-2):
        self._model = model
        self._model.eval()
        self._finalconv_name = finalconv_name
        self.__params = list(self.model.parameters())
        self._FCL_weight = np.squeeze(self.__params[FCL_index].data.numpy())
        self._data_transformer = data_transformer
        self.__features_blob = None
        self.__hook = self._model._modules.get(self._finalconv_name).register_forward_hook(self.hook_feature)
    def hookchange(self):
        self.__hook.remove()
        self.__hook = self._model._modules.get(self._finalconv_name).register_forward_hook(self.hook_feature)
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model
        sefl._model.eval()
        self.__params = list(self.model.parameters())
        self.hookchange()
    @property
    def finalconv_name(self):
        return self._finalconv_name
    @finalconv_name.setter
    def finalconv_name(self, name):
        self._finalconv_name = name
        self.hookchange()
    @property
    def FCL_weight(self):
        return self._FCL_weight
    @FCL_weight.setter
    def FCL_weight(self, FCL_index):
        self.FCL_weight = np.squeeze(self.__params[FCL_index].data.numpy())
    def hook_feature(self, module, input, output):
        self.__features_blob = output.data.cpu().numpy()
    def returnCAM(self, feature_conv, FCL_weight, class_idx):
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = np.dot(FCL_weight[idx], feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam -= np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img) 
            output_cam.append(cam_img)
        return output_cam
    def getHeatmap(self, img, heatmap_ratio=0.2, img_ratio=0.6):
        img_tensor = self._data_transformer(img)
        img_variable = Variable(img_tensor.unsqueeze(0))
        logit = self._model(img_variable)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        #getting heatmap
        CAMs = self.returnCAM(self.__features_blob, self._FCL_weight, [idx[0]])
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width, _ = opencvImage.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * heatmap_ratio + opencvImage * img_ratio
        return result
