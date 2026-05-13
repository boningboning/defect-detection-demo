# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset
from torchvision import transforms

import matplotlib.pyplot as plt

class yepian(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=2,
                 # num_classes=1,
                 multi_scale=True,
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 512),
                 scale_factor=16,
                 mean=[0.54654515,0.55549562,0.56890168],
                 std=[0.20032933,0.19896621,0.19539633],
                 bd_dilate_size=4):

        super(yepian, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {0:0, 1:1}
        self.class_weights = torch.FloatTensor([0.94237016,1.05762984]).cuda()
        # self.class_weights = torch.FloatTensor([1.0]).cuda()

        self.bd_dilate_size = bd_dilate_size
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 强制调整所有图像为 512x512
            transforms.ToTensor(),
        ])

    def save_edge(self, edge, sv_path, name):
        if isinstance(edge, torch.Tensor):
            edge = edge.cpu().numpy()

        edge = np.asarray(edge, dtype=np.uint8)
        edge[edge > 0] = 255

        save_img = Image.fromarray(edge)
        save_img.save(os.path.join(sv_path, name + '_edge.png'))

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        # temp = label.clone()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'yepian',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape


        # if 'test' in self.list_path:
        if self.list_path.endswith('test.lst'):
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name


        label = cv2.imread(os.path.join(self.root,'yepian',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)
        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            pred[pred > 0] = 255
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
