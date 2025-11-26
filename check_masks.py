import cv2 as cv
import numpy as np
import os

# 检查多个患者的mask标注
patients = ['1001', '1002', '1003', '1020', '1050']

for patient_id in patients:
    path = f'src/train/{patient_id}/arterial phase'
    if not os.path.exists(path):
        continue
        
    masks = [f for f in os.listdir(path) if 'mask.png' in f]
    total = 0
    non_empty = 0
    tumor_pixels_total = 0
    
    for mask_file in masks:
        mask_path = os.path.join(path, mask_file)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        tumor_pixels = np.count_nonzero(mask)
        total += 1
        if tumor_pixels > 0:
            non_empty += 1
            tumor_pixels_total += tumor_pixels
    
    print(f"\n患者 {patient_id}:")
    print(f"  总mask数: {total}")
    print(f"  有肿瘤标注: {non_empty}")
    print(f"  空mask: {total - non_empty}")
    if non_empty > 0:
        avg_tumor = tumor_pixels_total / non_empty
        print(f"  平均肿瘤像素: {avg_tumor:.0f} ({100*avg_tumor/262144:.2f}%)")
    else:
        print(f"  ⚠️ 该患者所有mask都是空的!")

# 检查整个训练集
print("\n" + "="*60)
print("扫描整个训练集...")
print("="*60)

all_patients = os.listdir('src/train')
total_masks = 0
total_non_empty = 0

for patient_id in all_patients:
    path = f'src/train/{patient_id}/arterial phase'
    if not os.path.exists(path):
        continue
        
    masks = [f for f in os.listdir(path) if 'mask.png' in f]
    for mask_file in masks:
        mask_path = os.path.join(path, mask_file)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        total_masks += 1
        if np.count_nonzero(mask) > 0:
            total_non_empty += 1

print(f"\n整体统计:")
print(f"  总mask数: {total_masks}")
print(f"  有肿瘤标注: {total_non_empty} ({100*total_non_empty/total_masks:.1f}%)")
print(f"  空mask: {total_masks - total_non_empty} ({100*(total_masks-total_non_empty)/total_masks:.1f}%)")
