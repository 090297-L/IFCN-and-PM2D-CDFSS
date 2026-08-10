import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


# 判断是否为图像文件
def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, nom_list=None , sub_list=None):
    # assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    image_label_list = []  
    list_read = open(data_list,encoding='UTF-8-sig').readlines()
    list_read2 = open(nom_list,encoding='UTF-8-sig').readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    nom_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []
        nom_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split()
        image_name = os.path.join(data_root, line_split[0])
        temp = line_split[0].replace('Images/', 'GT/')
        label_name = os.path.join(data_root, temp)
        label_name = label_name.replace('jpg', 'png')
        label_class = line_split[1]
        item = (image_name, label_name, label_class)

        image_label_list.append(item)

        sub_class_file_list[int(label_class)].append(item)
    for l_idx in tqdm(range(len(list_read2))):
        line = list_read2[l_idx]
        line = line.strip()
        line_split = line.split()
        image_nom = os.path.join(data_root, line_split[0])
        label_class1 = line_split[1]
        item = (image_nom, label_class1)

        nom_class_file_list[int(label_class1)].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list,nom_class_file_list


class SemData(Dataset):
    def __init__(self, split=2, shot=1, normal=1, data_root=None, data_list=None, nom_list=None, transform=None, mode='train',
                 use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.normal = normal
        self.data_root = data_root   

        # 获取训练类和测试类

        #self.class_list = list(range(1, 29))
        #if self.split == 3:
        #    self.sub_list = list(range(1, 22))
        #    self.sub_val_list = list(range(22, 29))
        #elif self.split == 2:
        #    self.sub_list = list(range(1, 15)) + list(range(22, 29))
        #    self.sub_val_list = list(range(15, 22))
        #elif self.split == 1:
        #    self.sub_list = list(range(1, 8)) + list(range(15, 29))
        #    self.sub_val_list = list(range(8, 15))
        #elif self.split == 0:
        #    self.sub_list = list(range(8, 29))
        #    self.sub_val_list = list(range(1, 8))

        # self.class_list = list(range(1, 13))
        # if self.split == 2:
        #     self.sub_list = list(range(1, 9))
        #     self.sub_val_list = list(range(9, 13))
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 5)) + list(range(9, 13))
        #     self.sub_val_list = list(range(5, 9))
        # elif self.split == 0:
        #     self.sub_list = list(range(5, 13))
        #     self.sub_val_list = list(range(1, 5))

        self.class_list = list(range(1, 25))
        if self.split == 2:
            self.sub_list = list(range(1, 17))
            self.sub_val_list = list(range(17, 25))
        elif self.split == 1:
            self.sub_list = list(range(1, 9)) + list(range(17, 25))
            self.sub_val_list = list(range(9, 17))
        elif self.split == 0:
            self.sub_list = list(range(9, 25))
            self.sub_val_list = list(range(1, 9))

        # self.class_list = list(range(1, 13))
        # if self.split == 3:
        #     self.sub_list = list(range(1, 10))
        #     self.sub_val_list = list(range(10, 13))
        # if self.split == 2:
        #     self.sub_list = list(range(1, 7)) + list(range(10, 13))
        #     self.sub_val_list = list(range(7, 10))
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 4)) + list(range(7, 13))
        #     self.sub_val_list = list(range(4, 7))
        # elif self.split == 0:
        #     self.sub_list = list(range(4, 13))
        #     self.sub_val_list = list(range(1, 4))

        # self.class_list = list(range(1, 16))
        # if self.split == 2:
        #     self.sub_list = list(range(1, 11))
        #     self.sub_val_list = list(range(11, 16))
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 6)) + list(range(11, 16))
        #     self.sub_val_list = list(range(6, 11))
        # elif self.split == 0:
        #     self.sub_list = list(range(6, 16))
        #     self.sub_val_list = list(range(1, 6))

        # self.class_list = list(range(1, 74))
        # if self.split == 14:
        #     self.sub_list = list(range(1, 74))
        #     self.sub_val_list = list(range(67, 74))# 7
        # if self.split == 13:
        #     self.sub_list = list(range(1, 62)) + list(range(67, 74))
        #     self.sub_val_list = list(range(62, 67))# 5
        # if self.split == 12:
        #     self.sub_list = list(range(1, 58)) + list(range(62, 74))
        #     self.sub_val_list = list(range(58, 62))# 4
        # if self.split == 11:
        #     self.sub_list = list(range(1, 57)) + list(range(58, 74))
        #     self.sub_val_list = list(range(57, 58))# 1
        # if self.split == 10:
        #     self.sub_list = list(range(1, 52)) + list(range(57, 74))
        #     self.sub_val_list = list(range(52, 57))# 5
        # if self.split == 9:
        #     self.sub_list = list(range(1, 47)) + list(range(52, 74))
        #     self.sub_val_list = list(range(47, 52))# 5
        # if self.split == 8:
        #     self.sub_list = list(range(1, 40)) + list(range(47, 74))
        #     self.sub_val_list = list(range(40, 47))# 7
        # if self.split == 7:
        #     self.sub_list = list(range(1, 36)) + list(range(40, 74))
        #     self.sub_val_list = list(range(36, 40))# 4
        # if self.split == 6:
        #     self.sub_list = list(range(1, 31)) + list(range(36, 74))
        #     self.sub_val_list = list(range(31, 36))# 5
        # if self.split == 5:
        #     self.sub_list = list(range(1, 27)) + list(range(31, 74))
        #     self.sub_val_list = list(range(27, 31))# 4
        # if self.split == 4:
        #     self.sub_list = list(range(1, 22)) + list(range(27, 74))
        #     self.sub_val_list = list(range(22, 27))# 5
        # if self.split == 3:
        #     self.sub_list = list(range(1, 17)) + list(range(22, 74))
        #     self.sub_val_list = list(range(17, 22))# 5
        # if self.split == 2:
        #     self.sub_list = list(range(1, 12)) + list(range(17, 74))
        #     self.sub_val_list = list(range(12, 17))# 5
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 4)) + list(range(12, 74))
        #     self.sub_val_list = list(range(4, 12))# 8
        # elif self.split == 0:
        #     self.sub_list = list(range(4, 74))
        #     self.sub_val_list = list(range(1, 4))# 3

        # self.class_list = list(range(1, 56))
        # if self.split == 8:
        #     self.sub_list = list(range(1, 51)) + list(range(51, 56))
        #     self.sub_val_list = list(range(51, 56))# 5
        # if self.split == 7:
        #     self.sub_list = list(range(1, 46)) + list(range(51, 56))
        #     self.sub_val_list = list(range(46, 51))# 5
        # if self.split == 6:
        #     self.sub_list = list(range(1, 41)) + list(range(46, 56))
        #     self.sub_val_list = list(range(41, 46))# 5
        # if self.split == 5:
        #     self.sub_list = list(range(1, 36)) + list(range(41, 56))
        #     self.sub_val_list = list(range(36, 41))# 5
        # if self.split == 4:
        #     self.sub_list = list(range(1, 31)) + list(range(36, 56))
        #     self.sub_val_list = list(range(31, 36))# 5
        # if self.split == 3:
        #     self.sub_list = list(range(1, 26)) + list(range(31, 56))
        #     self.sub_val_list = list(range(26, 31))# 5
        # if self.split == 2:
        #     self.sub_list = list(range(1, 16)) + list(range(26, 56))
        #     self.sub_val_list = list(range(16, 26))# 10
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 7)) + list(range(16, 56))
        #     self.sub_val_list = list(range(7, 16))# 9
        # elif self.split == 0:
        #     self.sub_list = list(range(7, 56))
        #     self.sub_val_list = list(range(1, 7))# 6

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        # 加载数据列表和transform
        if self.mode == 'train':
            self.data_list, self.sub_class_file_list,self.nom_class_file_list= make_dataset(split, data_root, data_list,nom_list, self.sub_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)  ##判断是否所有类别都有样本
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list ,self.nom_class_file_list= make_dataset(split, data_root, data_list,nom_list, self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        # 读入image和mask
        image_path, label_path, label_class = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label[label != 255] = 0
        label[label == 255] = 1


        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          

        assert len(label_class) > 0
        class_chosen = label_class
        file_class_chosen = self.sub_class_file_list[int(class_chosen)]
        nom_class_chosen = self.nom_class_file_list[int(class_chosen)]
        num_file = len(file_class_chosen)
        num_nom = len(nom_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path

            while ((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file)-1
                support_image_path, support_label_path,_ = file_class_chosen[support_idx]

            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)


        normal_image_path_list = []
        normal_idx_list = []
        for k in range(self.normal):
            nom_idx = random.randint(1, num_nom) - 1
            image_nom_path, _ = nom_class_chosen[nom_idx]
            while (nom_idx in normal_idx_list):
                nom_idx = random.randint(1, num_nom)-1
                image_nom_path,_ = nom_class_chosen[nom_idx]

            normal_idx_list.append(nom_idx)
            normal_image_path_list.append(image_nom_path)
        support_image_list = []
        normal_image_list = []
        label1=[]
        support_label_list = []
        subcls_list = []
        # nom_image = cv2.imread(image_nom_path, cv2.IMREAD_COLOR)
        # nom_image = cv2.cvtColor(nom_image, cv2.COLOR_BGR2RGB)
        # nom_image = np.float32(nom_image)
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(int(class_chosen)))
            else:
                subcls_list.append(self.sub_val_list.index(int(class_chosen)))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            support_label[support_label != 255] = 0
            support_label[support_label == 255] = 1

            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    

        for k in range(self.normal):
            normal_image_path = normal_image_path_list[k]
            normal_image = cv2.imread(normal_image_path, cv2.IMREAD_COLOR)
            normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
            normal_image = np.float32(normal_image)

            normal_image_list.append(normal_image)
            label1.append(label)

        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])

            for k in range(self.normal):
                normal_image_list[k], label1[k] = self.transform(normal_image_list[k], label1[k])

        nom =normal_image_list
        s_xs = support_image_list
        s_ys = support_label_list
        nom_s =nom[0]
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        for i in range(1, self.normal):
            nom_s = nom[i]+nom_s
        nom_s=nom_s/self.normal
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if self.mode == 'train':
            return image, label,nom_s, s_x, s_y, subcls_list
        else:
            return image, label,nom_s, s_x, s_y, subcls_list, raw_label