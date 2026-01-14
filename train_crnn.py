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
import re
from tqdm import tqdm

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 30
IMG_H = 32
IMG_W = 100
# 字符表：只有数字和字母
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    自增路径，例如: runs/train/exp --> runs/train/exp1, runs/train/exp2...
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        
        # 查找现有的 exp{n} 文件夹
        dirs = list(path.parent.glob(f"{path.name}{sep}*"))
        matches = [re.search(rf"%s{sep}(\d+)" % path.name, d.name) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2 # 如果没有数字，从2开始
        path = Path(f"{path}{sep}{n}{suffix}")
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def plot_results(csv_file, save_dir):
    """
    从 CSV 读取数据并画 Loss 图
    """
    try:
        if not os.path.exists(csv_file): return
        
        epochs = []
        train_losses = []
        val_losses = []
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                epochs.append(int(row[0]))
                train_losses.append(float(row[1]))
                val_losses.append(float(row[2]))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

# 1. 字符编码器
class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char2idx = {char: i + 1 for i, char in enumerate(alphabet)}
        self.idx2char = {i + 1: char for i, char in enumerate(alphabet)}

    def encode(self, text):
        text = text.lower()
        length = []
        result = []
        for item in text:
            if item in self.char2idx:
                result.append(self.char2idx[item])
        length.append(len(result))
        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

# 2. 数据集加载器
class OCRDataset(Dataset):
    def __init__(self, image_paths, converter):
        self.image_paths = image_paths
        self.converter = converter
        # 简单校验数据是否存在
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in the provided paths")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # 异常处理：读取失败生成黑图
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        filename = os.path.basename(path)
        # 兼容标签解析错误的情况
        try:
            label_text = filename.split('_')[0]
            encoded_label, label_len = self.converter.encode(label_text)
        except:
            encoded_label, label_len = self.converter.encode("error")

        return img, encoded_label, label_len

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    labels = torch.cat(labels, 0)
    lengths = torch.cat(lengths, 0)
    return imgs, labels, lengths

# 3. CRNN 模型
class CRNN(nn.Module):
    def __init__(self, n_class, input_channel=1, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=False),
        )
        self.embedding = nn.Linear(hidden_size * 2, n_class)

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv must be 1"
        conv = conv.squeeze(2).permute(2, 0, 1)
        output, _ = self.rnn(conv)
        T, b, h = output.size()
        output = self.embedding(output.view(T * b, h))
        output = output.view(T, b, -1)
        return output


def train():
    
    save_dir = increment_path(Path("runs/train/exp"), mkdir=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training started. Results will be saved to {save_dir}")
    
    # 初始化日志 CSV (增加 val_loss 列)
    csv_path = save_dir / "results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 准备转换器
    converter = LabelConverter(ALPHABET)
    n_class = len(ALPHABET) + 1 

    # 数据准备
    train_dir = 'dataset/train_crops'
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist.")
        return
    
    all_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')]
    if len(all_images) == 0:
        print("No images found.")
        return

    # 打乱并切分 90% 训练, 10% 验证
    np.random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.9)
    train_paths = all_images[:split_idx]
    val_paths = all_images[split_idx:]
    
    print(f"Total images: {len(all_images)} | Train: {len(train_paths)} | Val: {len(val_paths)}")

    # 创建 Dataset 和 DataLoader
    train_dataset = OCRDataset(train_paths, converter)
    val_dataset = OCRDataset(val_paths, converter)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate_fn)

    model = CRNN(n_class).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean') 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')

    # 训练循环
    print("Start Training...")
    for epoch in range(EPOCHS):
        # =========== TRAIN ===========
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
        
        for images, labels, label_lengths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            preds = model(images)
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=24, dtype=torch.long).to(device)
            log_probs = preds.log_softmax(2)
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad(): 
            for images, labels, label_lengths in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                preds = model(images)
                batch_size = images.size(0)
                input_lengths = torch.full(size=(batch_size,), fill_value=24, dtype=torch.long).to(device)
                log_probs = preds.log_softmax(2)
                
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Summary: Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        #保存日志
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
        
        # 绘制 Loss 曲线
        plot_results(csv_path, save_dir)

        # 保存模型
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }
        
        torch.save(ckpt, weights_dir / 'last.pt')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, weights_dir / 'best.pt')
            print(f"New best model found! Val Loss: {best_val_loss:.4f}")

    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()