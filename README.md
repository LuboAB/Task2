*注：为方便格式编辑，本文档使用markdown编辑，本文档的PDF版本由markdown转换而来。*

完整项目代码请通过此链接访问https://github.com/LuboAB/Task2

---

# 基于 EAST 与 CRNN 的自然场景文本检测与识别系统

## 一、 程序功能描述

本项目旨在实现一个完整的自然场景文本检测与识别（OCR）系统。面对自然场景（如街景招牌、广告牌）中背景复杂、光照不均、文字倾斜等挑战，系统采用了经典的**“两阶段”**技术路线：

1.  **文本检测（Text Detection）**：
    *   利用 **EAST (Efficient and Accurate Scene Text Detector)** 模型定位图像中的文本区域。
    *   该模块能够输出检测框的中心坐标、宽高以及旋转角度，从而有效处理倾斜文本。
2.  **图像校正与裁剪（Rectification & Cropping）**：
    *   根据检测到的旋转矩形信息，利用仿射变换（Affine Transformation）将倾斜的文本区域校正为水平方向，并从原图中裁剪出来，作为识别模型的输入。
3.  **文本识别（Text Recognition）**：
    *   利用 **CRNN (Convolutional Recurrent Neural Network)** 模型对裁剪后的区域进行序列识别。
    *   模型结合了 CNN 的特征提取能力和 RNN 的序列上下文建模能力，并通过 CTC (Connectionist Temporal Classification) 损失函数解决不定长序列对齐问题。

**系统整体流程：** 输入原始图片 → EAST检测 → NMS非极大值抑制 → 旋转框校正与裁剪 → CRNN推理 → 结果可视化。

---

## 二、 深度神经网络结构 (CRNN)

本项目的核心训练部分为 CRNN 识别网络。该网络结构设计精巧，主要由三部分组成：

1.  **卷积层 (Convolutional Layers) - 特征提取**
    *   采用了类似 VGG 的架构。
    *   **关键设计**：在第 3 和第 4 个池化层（MaxPooling）中，使用了 `kernel_size=(2, 2)` 但 `stride=(2, 1)` 的非对称设置。
    *   **目的**：将输入图像（固定高度32）的高度维度压缩至 1，但保留更多的宽度维度信息。这使得特征图在“时间轴”上具有足够的分辨率，以对应长文本序列。
    *   **输入**：1 × 32 × 100 (灰度图)
    *   **输出**：512 × 1 × 24 (特征序列)

2.  **循环层 (Recurrent Layers) - 序列建模**
    *   使用 **双向长短期记忆网络 (BiLSTM)**。
    *   **目的**：CNN 提取的特征是相互独立的，而 BiLSTM 能够捕捉字符之间的上下文关系（例如，“q”后面大概率跟“u”）。双向结构同时利用了前向和后向的语义信息。
    *   **输入**：24 × Batch × 512
    *   **输出**：24 × Batch × 512

3.  **转录层 (Transcription Layer) - 结果预测**
    *   使用全连接层将 RNN 的输出映射到字符类别空间（本例中为 36 个字符 + 1 个 Blank 占位符）。
    *   结合 **CTC Loss**，允许网络在不需要字符级位置标注的情况下进行端到端训练。

---

## 三、 源代码

### 1. 数据预处理 (`prepare_icdar15.py`)
解析 ICDAR 2015 数据集，根据标注切分出文本小图用于训练 CRNN。

```python
import os
import cv2
import numpy as np

# ================= 配置路径 =================
data_root = 'ICDAR2015/' 
train_img_dir = os.path.join(data_root, 'ch4_training_images')
train_gt_dir = os.path.join(data_root, 'ch4_training_localization_transcription_gt')
output_dir = 'dataset/train_crops' # 输出路径
# ===========================================

def crop_and_save():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = os.listdir(train_img_dir)
    count = 0

    for img_file in img_files:
        if not img_file.lower().endswith(('.jpg', '.png')): continue

        # 1. 读取图片
        img_path = os.path.join(train_img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        # 2. 读取对应的 GT 文件
        gt_filename = 'gt_' + os.path.splitext(img_file)[0] + '.txt'
        gt_path = os.path.join(train_gt_dir, gt_filename)
        
        if not os.path.exists(gt_path): continue

        # 3. 解析 GT
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 9: continue
                
                label = ",".join(parts[8:])
                if label == '###': continue # 忽略模糊文本

                try:
                    coords = list(map(int, parts[:8]))
                    pts = np.array(coords).reshape((-1, 1, 2))
                    
                    # 4. 裁剪区域 (使用外接矩形)
                    x, y, w, h = cv2.boundingRect(pts)
                    # 边界检查
                    x, y = max(0, x), max(0, y)
                    w, h = min(img.shape[1] - x, w), min(img.shape[0] - y, h)
                    
                    if w <= 0 or h <= 0: continue
                    crop_img = img[y:y+h, x:x+w]

                    # 5. 保存处理
                    safe_label = "".join([c for c in label if c.isalnum()])
                    if len(safe_label) < 1: continue

                    save_name = f"{safe_label}_{count}.jpg"
                    cv2.imwrite(os.path.join(output_dir, save_name), crop_img)
                    count += 1
                except ValueError: continue

    print(f"Done! Total {count} images saved to {output_dir}")

if __name__ == "__main__":
    crop_and_save()
```

### 2. 模型训练 (`train_crnn.py`)
定义网络结构、Dataset 并执行训练循环。

```python
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from tqdm import tqdm

# 配置
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50
IMG_H = 32
IMG_W = 100
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz" # 字符集

# label转换器略（见项目代码）

# 数据集加载器
class OCRDataset(Dataset):
    def __init__(self, image_paths, converter):
        self.image_paths = image_paths
        self.converter = converter

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        
        # 预处理：Resize + 归一化
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        filename = os.path.basename(path)
        try:
            label_text = filename.split('_')[0]
            encoded_label, label_len = self.converter.encode(label_text)
        except:
            encoded_label, label_len = self.converter.encode("error") # 容错

        return img, encoded_label, label_len

# CRNN 网络结构
class CRNN(nn.Module):
    def __init__(self, n_class, input_channel=1, hidden_size=256):
        super(CRNN, self).__init__()
        # VGG Style CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), # 宽方向 stride=1
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), # 宽方向 stride=1
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        # RNN (BiLSTM)
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=False),
        )
        # Classifier
        self.embedding = nn.Linear(hidden_size * 2, n_class)

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(2, 0, 1) # [w, b, c] -> [Seq, Batch, Feature]
        output, _ = self.rnn(conv)
        T, b, h = output.size()
        output = self.embedding(output.view(T * b, h))
        output = output.view(T, b, -1)
        return output

# 训练主逻辑 (Train Loop) 详见完整工程
```

### 3. 联合推理 (`demo_ocr.py`)
加载训练好的模型进行测试。

```python
def main(image_path):
    print("1. 加载模型...")
    # 加载 EAST
    east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
    
    # 加载 CRNN
    n_class = len(ALPHABET) + 1
    crnn_net = CRNN(n_class).to(DEVICE)
    
    # 检查是否有保存的模型
    if not list(Path(CRNN_MODEL_PATH).parent.glob("best.pt")): 
         print("Warning: best.pt not found, please check path.")

    checkpoint = torch.load(CRNN_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    crnn_net.load_state_dict(state_dict)
    crnn_net.eval()
    
    converter = LabelConverter(ALPHABET)

    print(f"2. 读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None: return
    vis_img = image.copy()

    print("3. 文字检测 (EAST)...")
    boxes = detection_east(image, east_net)
    print(f"   -> 检测到 {len(boxes)} 个文本框")

    print("4. 文字识别 (CRNN)...")
    crops = []
    valid_boxes = []
    
    # 裁剪
    for box in boxes:
        crop = get_rotated_rect_crop(image, box)
        if crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0:
            crops.append(crop)
            valid_boxes.append(box)

    # 识别
    texts = recognize_crnn(crnn_net, converter, crops)

    # 5. 可视化
    for box, text in zip(valid_boxes, texts):
        # 绘制文本框
        box_points = cv2.boxPoints(box)
        box_points = np.int64(box_points) 
        cv2.drawContours(vis_img, [box_points], 0, (0, 255, 0), 2)
        
        # 绘制文字
        # 放在框的左上角
        origin = tuple(box_points[1]) 
        cv2.putText(vis_img, text, (origin[0], origin[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        print(f"   Box: {box[0]} | Text: {text}")

    cv2.imshow("Final Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## 四、 训练过程与结果分析

### 1. 实验环境与设置
*   **数据集**：从 ICDAR 2015 训练集中自动裁剪出的约 4500 张文本图像。
*   **数据集划分**：90% 训练集 (4020张)，10% 验证集 (447张)。
*   **训练参数**：Batch Size=64, LR=0.001, Epochs=50, Optimizer=Adam。
*   **硬件**：NVIDIA CUDA 加速。

### 2. 训练日志分析
程序运行可靠，根据控制台输出的日志（截取）：

```text
Total images: 4467 | Train: 4020 | Val: 447
Start Training...
...
Summary: Epoch 1 | Train Loss: 4.0189 | Val Loss: 3.5643
New best model found! Val Loss: 3.5643
...
Summary: Epoch 15 | Train Loss: 2.9683 | Val Loss: 3.1038
New best model found! Val Loss: 3.1038
...
Summary: Epoch 30 | Train Loss: 0.4972 | Val Loss: 1.6247
New best model found! Val Loss: 1.6247
...
Summary: Epoch 50 | Train Loss: 0.0018 | Val Loss: 2.0405
✅ Training completed. Best Val Loss: 1.6247
```

![image-20260115045029997](https://github.com/LuboAB/Task2/blob/main/assets/image-20260115045029997.png?raw=true)

![img](https://github.com/LuboAB/Task2/blob/main/assets/loss.png?raw=true)

**结果解读：**

1.  **收敛性**：训练开始时 Loss 约为 4.0，在前 25 个 Epoch 内迅速下降，证明模型有效地学习到了图像特征到字符序列的映射关系。
2.  **最优模型**：**第 30 个 Epoch** 取得了验证集上的最小 Loss (**1.6247**)，并在此时保存了 `best.pt` 模型。
3.  **过拟合现象**：观察日志发现，从 Epoch 30 到 Epoch 50，**训练 Loss (Train Loss)** 继续从 0.49 降至几乎为 0 (0.0018)，但 **验证 Loss (Val Loss)** 却从 1.62 反弹至 2.04。
    *   这表明模型在后期出现了明显的**过拟合 (Overfitting)** 现象，泛化能力下降。
    *   **应对措施**：我们在训练代码中加入了 `Model Checkpoint` 机制，只保存验证集表现最好的模型，因此最终获得的 `best.pt` 避开了过拟合阶段，保证了推理时的性能。

### 3. 系统运行效果
最后，结合训练好的 CRNN 模型与 EAST 检测模型，对自然场景图片进行测试，能够准确框选文字位置并识别内容（如下图所示逻辑）：

*   检测：成功定位倾斜文本框。
*   校正：将倾斜文本旋转水平。
*   识别：CRNN 准确输出了对应的英文和数字内容。

![image-20260115045207399](https://github.com/LuboAB/Task2/blob/main/assets/image-20260115045207399.png?raw=true)

