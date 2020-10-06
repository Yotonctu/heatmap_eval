import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, sys, glob
import cv2
import tqdm
from argparse import ArgumentParser
from train.plugin.OQA_model import OQA_model
from utils.cam_pytorch import CAM_Heatmap_generator
from utils.argparseAction import savedAction
from utils.load_model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def Argparser():
    parser = ArgumentParser()
    parser.add_argument('-m','--model_path', type=str)
    parser.add_argument('-fc','--final_conv', type=str)
    parser.add_argument('-i','--image_path', type=str)
    parser.add_argument('-s','--saved',action=savedAction)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = Argparser()
    model_path = args.model_path
    finalconv_name = args.final_conv
    image_path = args.image_path
    saved = args.saved
    if(saved):
        saved_folder = args.saved_folder
    model, transformer = OQA_loader(model_path) 
    heatmap_generator = CAM_Heatmap_generator(model, transformer, finalconv_name)
    for p in tqdm.tqdm(glob.glob(f'{image_path}/*')):
        filename = p.split('/')[-1]
        img_pil = Image.open(p).convert('RGB')
        heatmap_img = heatmap_generator.getHeatmap(img_pil)
        if(saved):
            cv2.imwrite(f'{saved_folder}/{filename}',heatmap_img)
    
