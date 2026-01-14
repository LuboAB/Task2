import os
import cv2
import numpy as np
import shutil

# ================= 配置路径 =================
data_root = 'ICDAR2015/' 
train_img_dir = os.path.join(data_root, 'ch4_training_images')
train_gt_dir = os.path.join(data_root, 'ch4_training_localization_transcription_gt')

# 输出裁剪后图片的保存路径
output_dir = 'dataset/train_crops'
# ===========================================

def crop_and_save():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图片文件
    img_files = os.listdir(train_img_dir)
    count = 0

    for img_file in img_files:
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue

        # 1. 读取图片
        img_path = os.path.join(train_img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        # 2. 读取对应的 GT 文件
        # 图片名: img_1.jpg -> GT名: gt_img_1.txt
        gt_filename = 'gt_' + os.path.splitext(img_file)[0] + '.txt'
        gt_path = os.path.join(train_gt_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {img_file}")
            continue

        # 3. 解析 GT
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                # 去除换行符并分割
                parts = line.strip().split(',')
                
                # ICDAR2015 格式: x1,y1,x2,y2,x3,y3,x4,y4,transcription
                if len(parts) < 9: continue
                
                # 拼接最后可能包含逗号的文本 (例如: "Coca, Cola")
                label = ",".join(parts[8:])
                
                # 忽略 ### 标记的模糊文本
                if label == '###':
                    continue

                # 解析坐标
                try:
                    coords = list(map(int, parts[:8]))
                except ValueError:
                    continue
                
                pts = np.array(coords).reshape((-1, 1, 2))

                # 4. 裁剪区域
                # 对于倾斜文本，直接取外接矩形可能会包含过多背景
                # 这里简单起见，使用外接矩形 (Bounding Rect)
                x, y, w, h = cv2.boundingRect(pts)
                
                # 边界检查
                x = max(0, x)
                y = max(0, y)
                w = min(img.shape[1] - x, w)
                h = min(img.shape[0] - y, h)
                
                if w <= 0 or h <= 0: continue

                crop_img = img[y:y+h, x:x+w]

                # 5. 保存
                safe_label = "".join([c for c in label if c.isalnum()])
                if len(safe_label) < 1: continue

                save_name = f"{safe_label}_{count}.jpg"
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, crop_img)
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count} text crops...")

    print(f"Done! Total {count} images saved to {output_dir}")

if __name__ == "__main__":
    crop_and_save()