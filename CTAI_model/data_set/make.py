import os

import SimpleITK as sitk
import cv2 as cv
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import random_split

train_data_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/data'
test_data_path = '../data/test/'


def get_person_files(data_path):
    # 数据结构
    # [[person_id,image,mask],[person_id,image,mask],..,]

    all = []
    dir_list = [data_path + i for i in os.listdir(data_path)]
    for dir in dir_list:
        person_id = dir.split('/')[-1]
        filename_list = []
        image_list, mask_list, = [], []
        # 所有数据跑
        temp = os.listdir(dir + '/arterial phase')
        filename_list.extend([dir + '/arterial phase/' + name for name in temp])
        for i in filename_list:
            if '.dcm' in i:
                image_list.append(i)
            if '_mask' in i:
                mask_list.append(i)

        all.append([person_id, image_list, mask_list])

    return all


def get_train_files(data_path, all, get_dice=False):
    image_list, mask_list, finish_list, id_list = [], [], [], []
    # id_list  先是病人id再是图片
    dir_list = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    filename_list = []
    for dir in dir_list:
        # 所有数据跑
        if all:
            arterial_path = os.path.join(dir, 'arterial phase')
            temp = os.listdir(arterial_path)
            filename_list.extend([os.path.join(arterial_path, name) for name in temp])
        if not all:
            filename_list.append(dir)
            # temp = os.listdir(dir)
            # filename_list.extend([dir + '/' + name for name in temp])

    for i in filename_list:
        if '.dcm' in i:
            image_list.append((i, i.split('/')[-3], i.split('/')[-1].replace('.dcm', '')))
        if '_mask' in i:
            mask_list.append(i)
        if 'finish' in i:
            finish_list.append(i)
    if get_dice:
        return image_list, mask_list, id_list
    else:
        return image_list


def data_in_one(inputdata):
    if not inputdata.any():
        return inputdata
    inputdata = (inputdata - inputdata.min()) / (inputdata.max() - inputdata.min())
    return inputdata


def get_dataset(data_path, have):
    global test_image, test_mask
    image_list, mask_list, image_data, mask_data = [], [], [], []

    image_list = get_train_files(data_path, all=True)
    for i in image_list:
        image = sitk.ReadImage(i[0])
        image_array = sitk.GetArrayFromImage(image)
        mask = i[0].replace('.dcm', '_mask.png')
        mask_array = cv.imread(mask, cv.IMREAD_GRAYSCALE)

        if have:
            if not mask_array.any():
                continue

        mask_array = data_in_one(mask_array)
        mask_tensor = torch.from_numpy(mask_array).float()
        j = i[0].split('/')[-1].replace('_mask.png', '')
        mask_data.append((j, mask_tensor))

        ROI_mask = np.zeros(shape=image_array.shape)
        ROI_mask_mini = np.zeros(shape=(1, 160, 100))
        ROI_mask_mini[0] = image_array[0][270:430, 200:300]
        ROI_mask_mini = data_in_one(ROI_mask_mini)
        ROI_mask[0][270:430, 200:300] = ROI_mask_mini[0]
        test_image = ROI_mask
        image_tensor = torch.from_numpy(ROI_mask).float()
        image_data.append((image_tensor, i[1], i[2]))

    return image_data, mask_data


def get_onlytest(data_path, have):
    global test_image, test_mask
    image_list, mask_list, image_data, mask_data = [], [], [], []

    image_list = get_train_files(data_path, all=True)
    for i in image_list:
        image = sitk.ReadImage(i[0])
        image_array = sitk.GetArrayFromImage(image)

        ROI_mask = np.zeros(shape=image_array.shape)
        ROI_mask_mini = np.zeros(shape=(1, 160, 100))
        ROI_mask_mini[0] = image_array[0][270:430, 200:300]
        ROI_mask_mini = data_in_one(ROI_mask_mini)
        ROI_mask[0][270:430, 200:300] = ROI_mask_mini[0]
        test_image = ROI_mask
        image_tensor = torch.from_numpy(ROI_mask).float()
        # print(image_tensor.shape)
        image_data.append((image_tensor, i[1], i[2]))

    return image_data


class Dataset(data.Dataset):
    def __init__(self, path, have=True, transform=None):
        imgs = get_dataset(data_path=path, have=have)
        self.imgs = imgs
        # self.transform = transform
        # self.target_transform = target_transform

    def __getitem__(self, index):
        image = self.imgs[0][index]
        mask = self.imgs[1][index]

        return image, mask

    def __len__(self):
        return len(self.imgs[0])


class testDataset(data.Dataset):
    def __init__(self, path, have=True, transform=None):
        imgs = get_onlytest(data_path=path, have=have)
        self.imgs = imgs
        # self.transform = transform
        # self.target_transform = target_transform

    def __getitem__(self, index):
        image = self.imgs[index]
        return image

    def __len__(self):
        return len(self.imgs)


def get_d1(path, random_seed=42, train_transform=None, val_transform=None):
    """
    按患者ID划分训练集、验证集和测试集 (6:2:2)
    确保同一患者的所有切片只出现在一个集合中,避免数据泄露
    
    Args:
        path: 数据路径
        random_seed: 随机种子,确保可复现性
        train_transform: 训练集数据增强
        val_transform: 验证集数据增强 (通常为None)
    """
    import random
    import json
    
    # 获取所有患者ID并排序
    patient_ids = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    
    # 设置随机种子并打乱患者ID
    random.seed(random_seed)
    shuffled_patient_ids = patient_ids.copy()
    random.shuffle(shuffled_patient_ids)
    
    # 按患者ID划分 (60%训练, 20%验证, 20%测试)
    n_total = len(shuffled_patient_ids)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    
    train_patient_ids = sorted(shuffled_patient_ids[:n_train])
    val_patient_ids = sorted(shuffled_patient_ids[n_train:n_train + n_val])
    test_patient_ids = sorted(shuffled_patient_ids[n_train + n_val:])
    
    # 保存数据划分信息
    split_info = {
        'random_seed': random_seed,
        'total_patients': n_total,
        'split_ratio': '6:2:2',
        'train_patients': train_patient_ids,
        'val_patients': val_patient_ids,
        'test_patients': test_patient_ids,
        'train_count': len(train_patient_ids),
        'val_count': len(val_patient_ids),
        'test_count': len(test_patient_ids)
    }
    
    # 保存到文件
    split_file = os.path.join(os.path.dirname(path), 'data_split_info.json')
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"总患者数: {n_total}")
    print(f"训练患者: {len(train_patient_ids)} (随机划分,已保存到 {split_file})")
    print(f"验证患者: {len(val_patient_ids)}")
    print(f"测试患者: {len(test_patient_ids)}")
    
    # 分别加载训练、验证和测试数据
    train_dataset = PatientDataset(path, train_patient_ids, have=True, transform=train_transform)
    val_dataset = PatientDataset(path, val_patient_ids, have=True, transform=val_transform)
    test_dataset = PatientDataset(path, test_patient_ids, have=True, transform=None)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


class PatientDataset(data.Dataset):
    """按患者ID加载数据集"""
    def __init__(self, path, patient_ids, have=True, transform=None):
        self.images = []
        self.masks = []
        self.transform = transform
        
        for patient_id in patient_ids:
            patient_path = os.path.join(path, patient_id, 'arterial phase')
            if not os.path.exists(patient_path):
                continue
                
            files = os.listdir(patient_path)
            for filename in files:
                if '.dcm' in filename and '_mask' not in filename:
                    dcm_path = os.path.join(patient_path, filename)
                    mask_path = dcm_path.replace('.dcm', '_mask.png')
                    
                    # 读取图像
                    image = sitk.ReadImage(dcm_path)
                    image_array = sitk.GetArrayFromImage(image)
                    
                    # 读取mask
                    if os.path.exists(mask_path):
                        mask_array = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                    else:
                        mask_array = np.zeros(image_array.shape[1:])
                    
                    # 如果要求必须有标注,跳过空mask
                    if have and not mask_array.any():
                        continue
                    
                    # 预处理图像
                    ROI_mask = np.zeros(shape=image_array.shape)
                    ROI_mask_mini = np.zeros(shape=(1, 160, 100))
                    ROI_mask_mini[0] = image_array[0][270:430, 200:300]
                    ROI_mask_mini = data_in_one(ROI_mask_mini)
                    ROI_mask[0][270:430, 200:300] = ROI_mask_mini[0]
                    
                    image_tensor = torch.from_numpy(ROI_mask).float()
                    
                    # 预处理mask
                    mask_array = data_in_one(mask_array)
                    mask_tensor = torch.from_numpy(mask_array).float()
                    
                    # 保存
                    self.images.append((image_tensor, patient_id, filename.replace('.dcm', '')))
                    self.masks.append((filename.replace('.dcm', ''), mask_tensor))
    
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        
        # 应用数据增强
        if self.transform is not None:
            image_tensor = image[0]  # (1, 512, 512)
            mask_tensor = mask[1]    # (512, 512)
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
            # 重新组合为原始格式
            image = (image_tensor, image[1], image[2])
            mask = (mask[0], mask_tensor)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)


def get_d1_local(path):
    bag = testDataset(path, have=False)
    # train_size = int(0.9 * len(bag))
    # test_size = len(bag) - train_size
    # train_dataset, test_dataset = random_split(bag, [train_size, test_size])
    return bag


if __name__ == '__main__':
    # get_train_files(train_data_path)
    # get_dataset(train_data_path,have=True)
    bag = get_d1_local()
