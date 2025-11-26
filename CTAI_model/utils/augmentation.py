"""
医学CT图像数据增强模块
针对肝脏肿瘤分割任务设计
"""

import torch
import numpy as np
import cv2
import random

# scipy是可选依赖，用于弹性形变
try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Compose:
    """组合多个数据增强操作"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            # image: (1, 512, 512), mask: (512, 512)
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        return image, mask


class RandomVerticalFlip:
    """随机垂直翻转"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        return image, mask


class RandomRotation:
    """随机旋转 (-degrees, +degrees)"""
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # 转换为numpy进行旋转
            img_np = image.squeeze(0).numpy()  # (512, 512)
            mask_np = mask.numpy()
            
            h, w = img_np.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            img_np = cv2.warpAffine(img_np, M, (w, h), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
            mask_np = cv2.warpAffine(mask_np, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            
            image = torch.from_numpy(img_np).unsqueeze(0).float()
            mask = torch.from_numpy(mask_np).float()
        
        return image, mask


class RandomTranslation:
    """随机平移"""
    def __init__(self, translate=(0.1, 0.1), p=0.5):
        """
        Args:
            translate: (tx, ty) 最大平移比例 (相对于图像尺寸)
            p: 应用概率
        """
        self.translate = translate
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            img_np = image.squeeze(0).numpy()
            mask_np = mask.numpy()
            
            h, w = img_np.shape
            tx = int(random.uniform(-self.translate[0], self.translate[0]) * w)
            ty = int(random.uniform(-self.translate[1], self.translate[1]) * h)
            
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            img_np = cv2.warpAffine(img_np, M, (w, h),
                                    borderMode=cv2.BORDER_REFLECT)
            mask_np = cv2.warpAffine(mask_np, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            
            image = torch.from_numpy(img_np).unsqueeze(0).float()
            mask = torch.from_numpy(mask_np).float()
        
        return image, mask


class RandomBrightnessContrast:
    """随机调整亮度和对比度"""
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), p=0.5):
        """
        Args:
            brightness_range: 亮度调整范围 (乘法因子)
            contrast_range: 对比度调整范围 (乘法因子)
            p: 应用概率
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            # 亮度调整
            brightness = random.uniform(*self.brightness_range)
            image = image * brightness
            
            # 对比度调整
            contrast = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast + mean
            
            # 裁剪到[0, 1]
            image = torch.clamp(image, 0, 1)
        
        return image, mask


class RandomGaussianNoise:
    """添加高斯噪声"""
    def __init__(self, std_range=(0.01, 0.05), p=0.3):
        """
        Args:
            std_range: 噪声标准差范围
            p: 应用概率
        """
        self.std_range = std_range
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(image) * std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image, mask


class ElasticTransform:
    """弹性形变 - 医学图像常用增强"""
    def __init__(self, alpha=100, sigma=10, p=0.3):
        """
        Args:
            alpha: 形变强度
            sigma: 高斯滤波器标准差
            p: 应用概率
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            img_np = image.squeeze(0).numpy()
            mask_np = mask.numpy()
            
            shape = img_np.shape
            
            # 生成随机位移场
            dx = gaussian_filter((random.random() * 2 - 1) * np.ones(shape), 
                                 self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((random.random() * 2 - 1) * np.ones(shape), 
                                 self.sigma, mode="constant", cval=0) * self.alpha
            
            # 创建网格
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
            
            # 应用形变
            img_np = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
            mask_np = map_coordinates(mask_np, indices, order=0, mode='constant', cval=0).reshape(shape)
            
            image = torch.from_numpy(img_np).unsqueeze(0).float()
            mask = torch.from_numpy(mask_np).float()
        
        return image, mask


class RandomGaussianBlur:
    """随机高斯模糊"""
    def __init__(self, kernel_size=(3, 7), sigma_range=(0.1, 2.0), p=0.2):
        """
        Args:
            kernel_size: 核大小范围 (必须是奇数)
            sigma_range: 高斯标准差范围
            p: 应用概率
        """
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            img_np = image.squeeze(0).numpy()
            
            # 随机选择奇数核大小
            k = random.randrange(self.kernel_size[0], self.kernel_size[1] + 1, 2)
            if k % 2 == 0:
                k += 1
            
            sigma = random.uniform(*self.sigma_range)
            img_np = cv2.GaussianBlur(img_np, (k, k), sigma)
            
            image = torch.from_numpy(img_np).unsqueeze(0).float()
        
        return image, mask


def get_training_augmentation(mode='medium'):
    """
    获取训练时的数据增强组合
    
    Args:
        mode: 'light' - 轻度增强 (保守)
              'medium' - 中度增强 (推荐)
              'heavy' - 重度增强 (激进)
    
    Returns:
        Compose对象
    """
    if mode == 'light':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10, p=0.3),
            RandomBrightnessContrast(brightness_range=(0.9, 1.1), 
                                     contrast_range=(0.9, 1.1), p=0.3),
        ])
    
    elif mode == 'medium':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=15, p=0.4),
            RandomTranslation(translate=(0.05, 0.05), p=0.3),
            RandomBrightnessContrast(brightness_range=(0.8, 1.2),
                                     contrast_range=(0.8, 1.2), p=0.5),
            RandomGaussianNoise(std_range=(0.01, 0.03), p=0.2),
            RandomGaussianBlur(p=0.15),
        ])
    
    elif mode == 'heavy':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=20, p=0.5),
            RandomTranslation(translate=(0.1, 0.1), p=0.4),
            ElasticTransform(alpha=100, sigma=10, p=0.3),
            RandomBrightnessContrast(brightness_range=(0.7, 1.3),
                                     contrast_range=(0.7, 1.3), p=0.6),
            RandomGaussianNoise(std_range=(0.01, 0.05), p=0.3),
            RandomGaussianBlur(p=0.2),
        ])
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'light', 'medium', 'heavy'")


if __name__ == '__main__':
    # 测试代码
    print("数据增强模块测试")
    print("="*60)
    
    # 创建测试数据
    test_image = torch.rand(1, 512, 512)
    test_mask = torch.randint(0, 2, (512, 512)).float()
    
    # 测试各种增强
    transforms = [
        ('水平翻转', RandomHorizontalFlip(p=1.0)),
        ('垂直翻转', RandomVerticalFlip(p=1.0)),
        ('旋转15度', RandomRotation(degrees=15, p=1.0)),
        ('平移', RandomTranslation(p=1.0)),
        ('亮度对比度', RandomBrightnessContrast(p=1.0)),
        ('高斯噪声', RandomGaussianNoise(p=1.0)),
        ('高斯模糊', RandomGaussianBlur(p=1.0)),
        ('弹性形变', ElasticTransform(p=1.0)),
    ]
    
    for name, transform in transforms:
        img, msk = transform(test_image.clone(), test_mask.clone())
        print(f"✓ {name}: image shape={img.shape}, mask shape={msk.shape}")
    
    print("\n测试组合增强:")
    for mode in ['light', 'medium', 'heavy']:
        aug = get_training_augmentation(mode)
        img, msk = aug(test_image.clone(), test_mask.clone())
        print(f"✓ {mode}模式: image shape={img.shape}, mask shape={msk.shape}")
    
    print("\n✅ 所有测试通过!")
