import cv2
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ================= 配置区域 =================

# 视频存放的根目录
BASE_VIDEO_PATH = "/home/pzy/milmer/code/face_video"

# 截取出的图片存放根目录
OUTPUT_BASE_PATH = "/home/pzy/LibEER/LibEER/faces_224"

# 图像目标尺寸 (HSEmotion/EfficientNet 标准输入)
TARGET_SIZE = (224, 224)

# DEAP 数据集参数 (和之前保持一致)
FS = 128
VIDEO_OFFSET_SEC = 3.0  # 跳过前3秒基线

# 训练参数 (EEGNet/RGNN 通用设置: 1秒1张)
TRAIN_SAMPLE_LENGTH_POINTS = 128
TRAIN_STRIDE_POINTS = 128

SAMPLE_LENGTH_SEC = TRAIN_SAMPLE_LENGTH_POINTS / FS
STRIDE_SEC = TRAIN_STRIDE_POINTS / FS
TRIAL_DURATION_SEC = 60.0

# ===========================================

def get_face_detector():
    """初始化 MediaPipe 人脸检测器"""
    mp_face_detection = mp.solutions.face_detection
    # model_selection=1 适用于距离相机2-5米的人脸(全景)，0 适用于近距离(自拍/网络摄像头)
    # DEAP 数据集是被试看着屏幕，距离较近，建议用 0
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    return detector

def extract_frames():
    detector = get_face_detector()
    
    # 遍历32个被试
    for sub_id in range(1, 33):
        sub_str = f"s{sub_id:02d}"
        video_dir = os.path.join(BASE_VIDEO_PATH, sub_str)
        output_sub_dir = os.path.join(OUTPUT_BASE_PATH, sub_str)
        
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        print(f"Processing Subject: {sub_str}...")

        # 遍历40个Trial
        for trial_id in tqdm(range(1, 41), desc=f"Subject {sub_str}"):
            trial_str = f"trial{trial_id:02d}"
            video_filename = f"{sub_str}_{trial_str}.avi"
            video_path = os.path.join(video_dir, video_filename)
            
            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 计算 Segment 数量
            num_segments = int((TRIAL_DURATION_SEC - SAMPLE_LENGTH_SEC) // STRIDE_SEC) + 1
            
            for seg_idx in range(num_segments):
                # 1. 计算时间戳
                eeg_start_time = seg_idx * STRIDE_SEC
                target_time = VIDEO_OFFSET_SEC + eeg_start_time
                target_frame_idx = int(target_time * fps)
                
                if target_frame_idx >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # ================= 人脸检测与裁剪核心逻辑 =================
                    # MediaPipe 需要 RGB 输入
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.process(frame_rgb)
                    
                    face_img = None
                    
                    if results.detections:
                        # 默认取第一个检测到的人脸 (DEAP通常只有一个人)
                        detection = results.detections[0]
                        bboxC = detection.location_data.relative_bounding_box
                        
                        # 转换相对坐标为绝对像素坐标
                        x = int(bboxC.xmin * width)
                        y = int(bboxC.ymin * height)
                        w = int(bboxC.width * width)
                        h = int(bboxC.height * height)
                        
                        # 边界保护 (防止裁出负数或超出图片)
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        # 稍微扩大一点裁剪框，因为情绪模型通常需要完整的下巴和额头
                        # 这里的 padding 可以根据实际效果微调
                        padding = int(w * 0.1) 
                        x_p = max(0, x - padding)
                        y_p = max(0, y - padding)
                        w_p = min(width - x_p, w + 2 * padding)
                        h_p = min(height - y_p, h + 2 * padding)

                        if w_p > 0 and h_p > 0:
                            face_img = frame[y_p:y_p+h_p, x_p:x_p+w_p]
                    
                    # ================= 保存逻辑 =================
                    output_filename = f"{sub_str}_{trial_str}_seg{seg_idx:03d}.jpg"
                    output_full_path = os.path.join(output_sub_dir, output_filename)
                    
                    if face_img is not None:
                        # Resize 到 224x224
                        face_resized = cv2.resize(face_img, TARGET_SIZE)
                        cv2.imwrite(output_full_path, face_resized)
                    else:
                        # 如果没检测到人脸，为了保持数据对齐，建议做以下二选一：
                        # 1. 保存全黑图片 (推荐，保证 Dataset 加载时不报错)
                        # 2. 直接 Resize 原图 (不推荐，包含大量背景噪声)
                        # print(f"No face detected in {video_filename} at {target_time}s")
                        black_img = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
                        cv2.imwrite(output_full_path, black_img)

            cap.release()

if __name__ == "__main__":
    extract_frames()
    print("Face Extraction complete.")