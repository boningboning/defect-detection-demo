# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image
import datasets.cityscapes_labels as cityscapes_labels
# from cityscapes_labels import trainId2trainId as cityscapes_labels
import torch
from .base_dataset import BaseDataset
trainid_to_trainid = cityscapes_labels.color2trainId
color_to_trainid = cityscapes_labels.color2trainId
class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {34: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
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
        if 'train' in self.list_path:

            for item in self.img_list:
                if len(item) == 2:
                    image_path, label_path = item

                else:
                    image_path, label_path= item[0], item[1]
                # image_path, label_path = item
                # image_path, label_path, domain = item[0], item[1], int(item[2])
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                })
        if 'val' in self.list_path:
            for item in self.img_list:
                # image_path, label_path = item
                image_path, label_path = item[0], item[1]
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        for k, v in self.label_mapping.items():
            label[temp == k] = v

        # 映射完成后检测非法像素值
        valid_labels = set(range(19)) | {255}  # 0-18 + 255
        unique_labels = set(np.unique(label))
        invalid_labels = unique_labels - valid_labels
        if len(invalid_labels) > 0:
            # print(f"[Warning] Found invalid label values: {invalid_labels}")
            # 也可以改为抛异常
            raise ValueError(f"Invalid label values found: {invalid_labels}")

        return label

    # def convert_label(self, label, inverse=False):
    #     temp = label.copy()
    #     for k, v in self.label_mapping.items():
    #         label[temp == k] = v
    #     return label
    # def convert_label(self, label, inverse=False):
    #     temp = label.copy()
    #     for k, v in color_to_trainid.items():
    #         label[(temp == np.array(k))[:, :, 0] & (temp == np.array(k))[:, :, 1] & (temp == np.array(k))[:, :,2]] = v
    #
    #     return label
    def convert_label_val(self, label):
        temp = label.copy()
        temp = np.ones((temp.shape[0], temp.shape[1]), dtype=np.uint8) * 255
        for k, v in trainid_to_trainid.items():
            mask1 = np.all(label == k, axis=-1)
            temp[mask1] = v
        # label = Image.fromarray(temp.astype(np.uint8))
        return temp



    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        # # 获取基础域标签（从文件读取或映射）
        # base_domain = item["domain"]


        if 'test' in self.list_path:
            image = cv2.imread(os.path.join(self.root, 'GTAV', item["img"]),
                               cv2.IMREAD_COLOR)
            size = image.shape
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        if self.list_path.endswith('train.lst'):
            image = cv2.imread(os.path.join(self.root, 'GTAV', item["img"]),
                               cv2.IMREAD_COLOR)
            size = image.shape
            label_path = os.path.join(self.root, 'GTAV', item["label"])
            label = Image.open(label_path)  # 读取为 P 模式
            label = np.array(label)  # 保留类别索引，不解码为 RGB

            # label = cv2.imread(os.path.join(self.root, 'GTAV', item["label"]),
            #                    cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label)
            image, label, edge = self.gen_sample(image, label,
                                                     self.multi_scale, self.flip, edge_size=self.bd_dilate_size)


            return image.copy(),label.copy(),edge.copy(),np.array(size),name

        if self.list_path.endswith('val.lst'):
            image = cv2.imread(os.path.join(self.root, 'bdd-100k', item["img"]),
                               cv2.IMREAD_COLOR)
            size = image.shape
            label = cv2.imread(os.path.join(self.root, 'bdd-100k', item["label"]),
                               cv2.IMREAD_COLOR)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)  # OpenCV 默认 BGR，要转换成 RGB
            label = self.convert_label_val(label)
            # 验证集使用gen_sample处理
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
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
