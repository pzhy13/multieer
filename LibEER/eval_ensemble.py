import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
# 我们自己写切分逻辑，不依赖这个不稳定的 merge_to_part 了
from utils.args import get_args_parser
from utils.utils import setup_seed
from models.Models import Model

_original_load = torch.load
def safe_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_patch
from hsemotion.facial_emotions import HSEmotionRecognizer

class MultimodalDataset(Dataset):
    def __init__(self, eeg_data, eeg_labels, vis_paths, transform=None):
        self.eeg_data = eeg_data
        self.eeg_labels = eeg_labels
        self.vis_paths = vis_paths
        self.transform = transform
        self.empty_img = torch.zeros(3, 224, 224) 

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        lbl = self.eeg_labels[idx]
        if isinstance(lbl, np.ndarray) or isinstance(lbl, list):
            if len(lbl) > 1: target = torch.tensor(np.argmax(lbl), dtype=torch.long)
            else: target = torch.tensor(int(lbl), dtype=torch.long)
        else: target = torch.tensor(int(lbl), dtype=torch.long)
        
        img_path = self.vis_paths[idx]
        img = self.empty_img
        if img_path is not None and os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert('RGB')
                if self.transform: img = self.transform(pil_img)
            except: pass
        return eeg, img, target

def get_flat_visual_paths(subject_id, args, total_segments_expected):
    sub_str = f"s{subject_id:02d}"
    flat_paths = []
    
    # 动态计算每个 Trial 有多少个 segment
    num_segs_per_trial = int(total_segments_expected / 40) 
    
    for trial_id in range(1, 41):
        for seg_idx in range(num_segs_per_trial):
            candidates = [
                f"{sub_str}_trial{trial_id:02d}_seg{seg_idx+1}.jpg",
                f"{sub_str}_trial{trial_id:02d}_seg{seg_idx+1:03d}.jpg",
                f"{sub_str}_trial{trial_id:02d}_seg{seg_idx:03d}.jpg",
                f"{sub_str}_trial{trial_id:02d}_seg{seg_idx}.jpg",
            ]
            found_path = None
            for fname in candidates:
                fpath = os.path.join(args.faces_path, sub_str, fname)
                if os.path.exists(fpath):
                    found_path = fpath
                    break
            
            flat_paths.append(found_path)
            
    return flat_paths

# 🔥🔥🔥 新增：强制手动切分函数 🔥🔥🔥
def force_segmentation(raw_data, raw_label, sample_length, stride):
    segmented_data = []
    segmented_label = []
    
    print(f"Executing Manual Segmentation (Len={sample_length}, Stride={stride})...")
    
    for sub_idx, (sub_data, sub_lbls) in enumerate(zip(raw_data, raw_label)):
        # sub_data 应该是 List of Trials [Trial1, Trial2, ...]
        # 每个 Trial shape: (Channel, Time) e.g., (32, 7680)
        
        new_sub_data = []
        new_sub_label = []
        
        for trial_i, trial_eeg in enumerate(sub_data):
            # 确保是 (Channel, Time)
            if trial_eeg.shape[0] != 32: 
                # 有时候数据可能是 (Time, Channel)，需要转置，DEAP 通常是 32xTime
                pass 
                
            n_channels, n_points = trial_eeg.shape
            label = sub_lbls[trial_i]
            
            start = 0
            while start + sample_length <= n_points:
                seg = trial_eeg[:, start : start + sample_length]
                new_sub_data.append(seg)
                new_sub_label.append(label)
                start += stride
                
        segmented_data.append(np.array(new_sub_data))
        segmented_label.append(np.array(new_sub_label))
        
        if sub_idx == 0:
            print(f"  Example S01: Trial -> {len(new_sub_data)} segments. Seg Shape: {new_sub_data[0].shape}")

    return segmented_data, segmented_label

def main():
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True)
    parser.add_argument('-eeg_dir', type=str, required=True)
    parser.add_argument('-vis_dir', type=str, required=True)
    parser.add_argument('-alpha', type=float, default=0.6)
    args = parser.parse_args()

    # 这里的 only_seg 已经不重要了，因为我们会手动切
    # args.only_seg = True 

    if args.setting is not None: setting = preset_setting[args.setting](args)
    else: setting = set_setting_by_args(args)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Loading EEG Data structure...")
    
    # 1. 获取原始 Trial 级数据
    raw_data, raw_label, channels, feature_dim, num_classes = get_data(setting)
    
    # 2. 🔥🔥🔥 强制手动切分！🔥🔥🔥
    # 确保 sample_length 和 stride 被正确读取
    seg_len = args.sample_length if args.sample_length else 128
    seg_stride = args.stride if args.stride else 128
    
    raw_data, raw_label = force_segmentation(raw_data, raw_label, seg_len, seg_stride)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_acc = []
    
    for sub_idx, (data_sub, label_sub) in enumerate(zip(raw_data, raw_label), 1):
        # data_sub 已经是 numpy array (2400, 32, 128)
        
        print(f"\n========== Subject {sub_idx} Evaluation ==========")
        eeg_ckpt = os.path.join(args.eeg_dir, f"sub{sub_idx:02d}", "checkpoint-bestacc")
        vis_ckpt = os.path.join(args.vis_dir, f"visual_model_sub{sub_idx}_fold1_best.pth")
        
        split_path = os.path.join(args.eeg_dir, f"sub{sub_idx:02d}", "split.pkl")
        
        if not os.path.exists(split_path):
             print(f"❌ Split file missing: {split_path}. Skipping.")
             continue
        
        with open(split_path, 'rb') as f:
            tts = pickle.load(f)
        
        test_idx = tts['test'][0] 
        
        if not os.path.exists(eeg_ckpt):
            print(f"❌ EEG Checkpoint missing: {eeg_ckpt}")
            continue
        if not os.path.exists(vis_ckpt):
            print(f"❌ Visual Checkpoint missing: {vis_ckpt}")
            continue

        eeg_model = Model['EEGNet'](channels, feature_dim, num_classes).to(device)
        try:
            ckpt = torch.load(eeg_ckpt, map_location=device)
            eeg_model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
            eeg_model.eval()
        except Exception as e: 
            print(f"Error loading EEG model: {e}")
            continue

        fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
        vis_model = fer.model
        num_ftrs = 1280
        try:
            if hasattr(vis_model, 'classifier') and not isinstance(vis_model.classifier, nn.Identity): 
                num_ftrs = vis_model.classifier.in_features
            elif hasattr(vis_model, 'fc') and not isinstance(vis_model.fc, nn.Identity): 
                num_ftrs = vis_model.fc.in_features
        except: pass
        
        # 适配你的 256 层结构
        vis_model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(256, num_classes)
        )
        if hasattr(vis_model, 'fc'): vis_model.fc = nn.Identity()
        
        try:
            vis_model.load_state_dict(torch.load(vis_ckpt, map_location=device))
            vis_model.to(device).eval()
        except Exception as e: 
            print(f"Error loading Vis model: {e}")
            continue
        
        # 3.1 获取该被试所有的图片路径
        total_segments = data_sub.shape[0] 
        vis_paths_all = get_flat_visual_paths(sub_idx, args, total_segments)
        
        # 3.2 提取测试数据
        test_eeg = data_sub[test_idx]
        test_label = label_sub[test_idx]
        test_vis_paths = [vis_paths_all[i] for i in test_idx]

        dataset = MultimodalDataset(test_eeg, test_label, test_vis_paths, transform=test_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for eeg, img, target in tqdm(loader, desc=f"Inference S{sub_idx}", leave=False):
                eeg, img, target = eeg.to(device), img.to(device), target.to(device)
                out_eeg = eeg_model(eeg)
                out_vis = vis_model(img)
                prob_final = (1 - args.alpha) * torch.softmax(out_eeg, 1) + args.alpha * torch.softmax(out_vis, 1)
                correct += (prob_final.argmax(1) == target).sum().item()
                total += target.size(0)
                
        acc = correct / total
        print(f"👉 Subject {sub_idx} Fusion Acc: {acc:.4f}")
        total_acc.append(acc)

    print(f"\nAvg Fusion Acc: {np.mean(total_acc):.4f}")

if __name__ == '__main__':
    main()