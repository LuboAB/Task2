import cv2
import torch
import numpy as np
import math
from train_crnn import CRNN, LabelConverter, ALPHABET, IMG_H, IMG_W
from pathlib import Path 

EAST_MODEL_PATH = "frozen_east_text_detection.pb"
CRNN_MODEL_PATH = "runs/train/exp4/weights/best.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decode_predictions(text_batch_logits, converter):
    """
    CRNN 解码工具函数
    """
    preds_size = torch.IntTensor([text_batch_logits.size(0)] * text_batch_logits.size(1))
    _, preds_index = text_batch_logits.max(2)
    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
    
    # 转为字符串
    sim_preds = converter.decode(preds_index.data, preds_size.data, raw=False)
    return sim_preds

def get_rotated_rect_crop(image, rect):
    """
    裁剪旋转矩形区域
    rect: ((center_x, center_y), (width, height), angle)
    """
    center, size, angle = rect
    center, size = list(map(int, center)), list(map(int, size))
    
    # 限制尺寸防止报错
    if size[0] == 0 or size[1] == 0:
        return None

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # 执行仿射变换旋转图像
    img_rot = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    # 裁剪对其后的矩形
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    
    # 如果高度 > 宽度 (竖排文字)，旋转90度
    if size[1] > size[0]:
         img_crop = cv2.rotate(img_crop, cv2.ROTATE_90_CLOCKWISE)
         
    return img_crop

def detection_east(image, net):
    """
    运行 EAST 模型检测文字框
    """
    orig = image.copy()
    (H, W) = image.shape[:2]
    
    # 调整大小为 32 的倍数
    newW, newH = (1280, 736)
    rW = W / float(newW)
    rH = H / float(newH)
    
    image_resized = cv2.resize(image, (newW, newH))
    
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    [boxes, confidences] = decode_east(scores, geometry, 0.5)
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices:
            # 还原坐标到原图
            # box: ((cx, cy), (w, h), angle)
            box = boxes[i]
            center = (box[0][0] * rW, box[0][1] * rH)
            size = (box[1][0] * rW, box[1][1] * rH)
            angle = box[2]
            final_boxes.append((center, size, angle))
            
    return final_boxes

def decode_east(scores, geometry, scoreThresh):
    detections = []
    confidences = []
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            if score < scoreThresh: continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                       offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))
    return [detections, confidences]

def recognize_crnn(crnn_net, converter, crops):
    """
    批量识别裁剪出的图片
    """
    if not crops: return []
    
    # 预处理 Batch
    imgs_tensor = []
    for crop in crops:
        # 转灰度
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Resize 到 100x32
        resized = cv2.resize(gray, (IMG_W, IMG_H))
        normalized = resized.astype(np.float32) / 255.0
        imgs_tensor.append(torch.from_numpy(normalized).unsqueeze(0)) # [1, 32, 100]

    # 堆叠成 Batch [B, 1, 32, 100]
    batch_input = torch.stack(imgs_tensor, 0).to(DEVICE)
    
    # 推理
    with torch.no_grad():
        preds = crnn_net(batch_input) # [T, B, n_class]
        
    # 解码
    log_probs = preds.log_softmax(2)
    texts = decode_predictions(log_probs, converter)
    return texts

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

if __name__ == "__main__":
    test_img = "ICDAR2015/ch4_test_images/img_498.jpg"
    main(test_img)