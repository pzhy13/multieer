import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
# 🔥 新增导入 confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import argparse

# 引入 LibEER 工具
from config.setting import preset_setting, set_setting_by_args
from data_utils.split import get_split_index, merge_to_part
from utils.args import get_args_parser
from utils.utils import setup_seed
from data_utils.load_data import get_data

# 引入模型
from models.EEGNet import EEGNet
from models.RGNN import RGNN

# 🔴 HSEmotion 补丁 (防止部分环境中 pickle load 报错)
_original_load = torch.load
def safe_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_patch
from hsemotion.facial_emotions import HSEmotionRecognizer

# ================= 1. 损失函数 (BMCL 核心) =================

class CMDLoss(nn.Module):
    """ Central Moment Discrepancy (CMD) 用于拉近公共空间的分布 """
    def __init__(self, k=3):
        super(CMDLoss, self).__init__()
        self.k = k

    def forward(self, x1, x2):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        losses = []
        losses.append(torch.norm(mx1 - mx2, p=2))
        for i in range(2, self.k + 1):
            order_moment_x1 = torch.mean(sx1.pow(i), dim=0)
            order_moment_x2 = torch.mean(sx2.pow(i), dim=0)
            losses.append(torch.norm(order_moment_x1 - order_moment_x2, p=2))
        return sum(losses)

class DiffLoss(nn.Module):
    """ 差异性损失，确保公共特征和私有特征正交 """
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, common, private):
        batch_size = common.size(0)
        common_n = F.normalize(common, dim=1)
        private_n = F.normalize(private, dim=1)
        correlation_matrix = torch.mm(common_n.t(), private_n)
        return torch.norm(correlation_matrix, p='fro').pow(2) / (batch_size * batch_size)

# ================= 2. 耦合网络架构 =================

class CoupledModel(nn.Module):
    def __init__(self, visual_backbone_name, eeg_backbone_name, num_classes=4, 
                 common_dim=128, private_dim=64, 
                 eeg_time_dim=128, eeg_channels=32, freeze_vis=True):
        super(CoupledModel, self).__init__()
        
        # --- Visual Branch ---
        if visual_backbone_name == 'hsemotion':
            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            self.vis_net = fer.model
            
            # 默认 EfficientNet-B0 维度
            self.vis_feat_dim = 1280 
            
            # 安全检查并获取特征维度
            try:
                if hasattr(self.vis_net, 'classifier') and not isinstance(self.vis_net.classifier, nn.Identity):
                    c = self.vis_net.classifier
                    if isinstance(c, nn.Linear): 
                        self.vis_feat_dim = c.in_features
                    elif isinstance(c, nn.Sequential):
                        for m in c:
                            if isinstance(m, nn.Linear): 
                                self.vis_feat_dim = m.in_features
                                break
                    elif hasattr(c, 'in_features'):
                        self.vis_feat_dim = c.in_features
                        
                elif hasattr(self.vis_net, 'fc') and not isinstance(self.vis_net.fc, nn.Identity):
                    c = self.vis_net.fc
                    if hasattr(c, 'in_features'):
                        self.vis_feat_dim = c.in_features
            except Exception as e:
                print(f"Warning: Could not auto-detect visual feature dim ({e}), using default {self.vis_feat_dim}")

            # 替换为 Identity
            if hasattr(self.vis_net, 'classifier'): self.vis_net.classifier = nn.Identity()
            elif hasattr(self.vis_net, 'fc'): self.vis_net.fc = nn.Identity()

        elif visual_backbone_name == 'resnet':
            self.vis_net = models.resnet50(pretrained=True)
            self.vis_feat_dim = self.vis_net.fc.in_features
            self.vis_net.fc = nn.Identity()
            
        if freeze_vis:
            for param in self.vis_net.parameters():
                param.requires_grad = False
        
        # --- EEG Branch ---
        self.eeg_backbone_name = eeg_backbone_name
        if eeg_backbone_name == 'eegnet':
            self.eeg_net = EEGNet(num_electrodes=eeg_channels, datapoints=eeg_time_dim, num_classes=num_classes)
            # 计算 EEGNet Flatten 后的维度: F2 * (time // 32)
            # F1=8, D=2 => F2=16. Example: 128 // 32 = 4. 16 * 4 = 64.
            f2 = self.eeg_net.F1 * self.eeg_net.D
            flatten_dim = f2 * (eeg_time_dim // 32)
            self.eeg_feat_dim = flatten_dim
            self.eeg_net.fc = nn.Identity()
            
        elif eeg_backbone_name == 'rgnn':
            # RGNN 需要根据配置动态确定 hidden dim，这里假设默认 512
            self.eeg_net = RGNN(num_electrodes=eeg_channels, num_classes=num_classes, num_hidden=512)
            self.eeg_feat_dim = 512
            # 替换 RGNN 的分类层
            self.eeg_net.fc2 = nn.Identity()

        # --- Projectors (映射到 Common 和 Private 空间) ---
        self.vis_common_fc = nn.Sequential(nn.Linear(self.vis_feat_dim, common_dim), nn.BatchNorm1d(common_dim), nn.ReLU())
        self.vis_private_fc = nn.Sequential(nn.Linear(self.vis_feat_dim, private_dim), nn.BatchNorm1d(private_dim), nn.ReLU())
        
        self.eeg_common_fc = nn.Sequential(nn.Linear(self.eeg_feat_dim, common_dim), nn.BatchNorm1d(common_dim), nn.ReLU())
        self.eeg_private_fc = nn.Sequential(nn.Linear(self.eeg_feat_dim, private_dim), nn.BatchNorm1d(private_dim), nn.ReLU())

        # --- Classifiers ---
        # 融合分类器使用所有特征的拼接
        fusion_input_dim = (common_dim + private_dim) * 2
        
        self.dropout = nn.Dropout(0.5)
        
        # 辅助分类器，帮助单模态特征学习
        self.vis_classifier = nn.Linear(common_dim + private_dim, num_classes)
        self.eeg_classifier = nn.Linear(common_dim + private_dim, num_classes)
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, eeg):
        # Visual Feature Extraction
        vis_feat = self.vis_net(img)
        
        # EEG Feature Extraction
        if self.eeg_backbone_name == 'rgnn':
            if eeg.dim() == 4 and eeg.size(1) == 1: eeg = eeg.squeeze(1) 
            eeg_feat, _ = self.eeg_net(eeg) 
        else:
            # EEGNet
            eeg_feat = self.eeg_net(eeg)

        # Projections
        vis_c = self.vis_common_fc(vis_feat)
        vis_p = self.vis_private_fc(vis_feat)
        eeg_c = self.eeg_common_fc(eeg_feat)
        eeg_p = self.eeg_private_fc(eeg_feat)

        # Concatenate for auxiliary classifiers
        vis_combined = torch.cat([vis_c, vis_p], dim=1)
        eeg_combined = torch.cat([eeg_c, eeg_p], dim=1)
        
        # Classifiers
        out_vis = self.vis_classifier(self.dropout(vis_combined))
        out_eeg = self.eeg_classifier(self.dropout(eeg_combined))
        
        # Fusion
        fusion_feat = torch.cat([vis_c, vis_p, eeg_c, eeg_p], dim=1)
        out_fusion = self.fusion_classifier(self.dropout(fusion_feat))

        return {
            'vis_c': vis_c, 'vis_p': vis_p, 
            'eeg_c': eeg_c, 'eeg_p': eeg_p,
            'out_vis': out_vis, 'out_eeg': out_eeg, 'out_fusion': out_fusion
        }

# ================= 3. 数据集与对齐 =================

class MultimodalDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: List of tuples (img_path, eeg_tensor, label)
        """
        self.data_list = data_list 
        self.transform = transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        img_path, eeg_data, label_val = self.data_list[idx]
        
        # Load Image
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: 
                img = self.transform(img)
        except Exception as e: 
            # Fallback for missing images
            # print(f"Error loading {img_path}: {e}")
            img = torch.zeros(3, 224, 224)
            
        # EEG Tensor
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        
        return img, eeg_tensor, torch.tensor(label_val, dtype=torch.long)

def get_visual_paths_per_trial(args):
    """
    仿照 train_visual_model_aligned.py 的逻辑获取图片路径。
    返回结构: aligned_data[sub_idx][trial_idx] -> List[img_paths]
    """
    print(f"Indexing Visual Data from {args.faces_path}...")
    aligned_img_paths = []

    for sub_id in range(1, 33):
        sub_str = f"s{sub_id:02d}"
        sub_face_dir = os.path.join(args.faces_path, sub_str)
        
        sub_trails_paths = []
        
        # DEAP has 40 trials
        for trial_id in range(1, 41):
            pattern = f"{sub_str}_trial{trial_id:02d}_seg*.jpg"
            search_path = os.path.join(sub_face_dir, pattern)
            files = glob.glob(search_path)
            # 关键：排序逻辑必须与视觉脚本一致
            files.sort(key=lambda x: int(x.split('_seg')[-1].split('.')[0]))
            
            sub_trails_paths.append(files) # 即使为空也append，保持索引对应
            
        aligned_img_paths.append(sub_trails_paths)
        
    return aligned_img_paths

def prepare_multimodal_data(eeg_data, eeg_labels, img_paths_struct, indices, sub_idx):
    """
    根据 split indices (trial level) 展平数据为 samples。
    并对齐 EEG segment 和 Image segment。
    """
    dataset_list = []
    
    # eeg_data[sub_idx] 是 List of Trials, 每个 Trial 是 (NumSegs, Channels, Time)
    sub_eeg = eeg_data[sub_idx] 
    sub_lbl = eeg_labels[sub_idx]
    sub_imgs = img_paths_struct[sub_idx] # List of Trials, 每个 Trial 是 List of Paths
    
    for t_idx in indices:
        if t_idx >= len(sub_eeg): continue
        
        # 获取该 Trial 的所有 Segments
        trial_eeg = sub_eeg[t_idx] # Shape: (60, 32, 128)
        trial_lbl = sub_lbl[t_idx] # Shape: (60,)
        trial_imgs = sub_imgs[t_idx] # List of paths
        
        # 对齐检查：取两者最小长度，防止越界
        num_samples = min(len(trial_eeg), len(trial_imgs))
        
        if num_samples == 0:
            continue
            
        for s_idx in range(num_samples):
            # 获取数据
            eeg_sample = trial_eeg[s_idx]
            img_path = trial_imgs[s_idx]
            lbl_sample = trial_lbl[s_idx]
            
            # 处理 Label (如果是 One-hot 转 index)
            if isinstance(lbl_sample, np.ndarray) and lbl_sample.ndim > 0:
                label_val = np.argmax(lbl_sample)
            else:
                label_val = int(lbl_sample)
            
            dataset_list.append((img_path, eeg_sample, label_val))
            
    return dataset_list

# ================= 4. 辅助打印函数 =================
def print_metrics(targets, preds, classes=None):
    cm = confusion_matrix(targets, preds)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) # 防止除零
    macro_f1 = f1_score(targets, preds, average='macro')
    acc = accuracy_score(targets, preds)
    
    print(f"\n{'='*20} Metrics Report {'='*20}")
    print(f"Overall Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print("-" * 30)
    print(f"Confusion Matrix:\n{cm}")
    print("-" * 30)
    print("Per-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        cls_name = classes[i] if classes else f"Class {i}"
        print(f"  {cls_name}: {acc:.4f} ({cm[i,i]}/{cm.sum(axis=1)[i]})")
    print(f"{'='*56}\n")
    return acc, macro_f1

# ================= 5. 主程序 =================

def main(args):
    if args.output_dir is None: args.output_dir = "BMCL_Result"
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    # --- Setting 配置 ---
    if args.setting is None: setting = set_setting_by_args(args)
    else: setting = preset_setting[args.setting](args)

    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # --- 1. 加载 EEG 数据 ---
    print("Loading EEG Data via LibEER...")
    # get_data 返回 (Session, Subject, Trial...), 需要 merge_to_part 展平为 (Subject, Trial...)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data_all, label_all = merge_to_part(data, label, setting)
    
    # 假设 DEAP 4分类的含义 (Valence/Arousal 组合)
    # 0: LVLA (Low Valence Low Arousal), 1: LVHA, 2: HVLA, 3: HVHA (具体顺序取决于你的 label_process)
    class_names = [f"Class {i}" for i in range(num_classes)]
    
    print("Indexing Image Paths...")
    img_paths_all = get_visual_paths_per_trial(args)
    
    # --- 3. 定义数据增强 (与单模态视觉脚本保持一致) ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_accuracies = []

    # --- 4. 逐个被试训练 (Subject Dependent) ---
    for sub_idx in range(len(data_all)):
        sub_id = sub_idx + 1
        print(f"\n========== Processing Subject {sub_id:02d} ==========")
        
        # --- 读取 Split (严格复用) ---
        split_file = os.path.join(args.eeg_dir, f"sub{sub_id:02d}", "split.pkl")
        
        if os.path.exists(split_file):
            print(f"Loading split from: {split_file}")
            with open(split_file, 'rb') as f:
                tts = pickle.load(f)
        else:
            print(f"Warning: Split file not found at {split_file}. Using dynamic split.")
            setup_seed(args.seed + sub_id)
            tts = get_split_index(data_all[sub_idx], label_all[sub_idx], setting)

        # 遍历 Folds
        for fold_idx, (train_idx, test_idx, val_idx) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            if val_idx[0] == -1 or len(val_idx) == 0: val_idx = test_idx
            
            # --- 准备多模态数据 ---
            train_list = prepare_multimodal_data(data_all, label_all, img_paths_all, train_idx, sub_idx)
            val_list = prepare_multimodal_data(data_all, label_all, img_paths_all, val_idx, sub_idx)
            test_list = prepare_multimodal_data(data_all, label_all, img_paths_all, test_idx, sub_idx)
            
            if len(train_list) == 0:
                print(f"Skipping S{sub_id} Fold {fold_idx} (No data)")
                continue

            # DataLoader
            train_loader = DataLoader(MultimodalDataset(train_list, transform=train_transform), 
                                      batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(MultimodalDataset(val_list, transform=val_test_transform), 
                                    batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(MultimodalDataset(test_list, transform=val_test_transform), 
                                     batch_size=args.batch_size, shuffle=False, num_workers=4)

            # --- 初始化模型 ---
            model = CoupledModel(
                visual_backbone_name=args.vis_backbone,
                eeg_backbone_name=args.eeg_backbone,
                num_classes=num_classes,
                eeg_time_dim=feature_dim,
                eeg_channels=channels,
                freeze_vis=True
            ).to(device)
            
            # 优化器
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=args.lr, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            
            crit_cls = nn.CrossEntropyLoss()
            crit_cmd = CMDLoss()
            crit_diff = DiffLoss()
            
            best_val_score = 0.0 # 根据 F1 还是 Acc 保存？通常 F1 更稳健，这里保持 Acc 以兼容之前逻辑，也可改为 F1
            best_f1_score = 0.0
            best_model_path = os.path.join(args.output_dir, f"sub{sub_id:02d}", "checkpoint-bestacc.pth")
            best_model_path_f1 = os.path.join(args.output_dir, f"sub{sub_id:02d}", "checkpoint-bestf1.pth")
            if not os.path.exists(os.path.dirname(best_model_path)): os.makedirs(os.path.dirname(best_model_path))

            pbar = tqdm(range(args.epochs), desc=f"S{sub_id}-Fold{fold_idx}", leave=False)
            for epoch in pbar:
                model.train()
                train_loss_sum = 0
                train_preds, train_targets = [], []
                
                for imgs, eegs, lbls in train_loader:
                    imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                    optimizer.zero_grad()
                    out = model(imgs, eegs)
                    
                    l_cls = crit_cls(out['out_fusion'], lbls) + 0.5 * crit_cls(out['out_vis'], lbls) + 0.5 * crit_cls(out['out_eeg'], lbls)
                    l_sim = args.alpha * crit_cmd(out['vis_c'], out['eeg_c'])
                    l_diff = args.beta * (crit_diff(out['vis_c'], out['vis_p']) + crit_diff(out['eeg_c'], out['eeg_p']))
                    
                    loss = l_cls + l_sim + l_diff
                    loss.backward()
                    optimizer.step()
                    
                    train_loss_sum += loss.item()
                    train_preds.extend(out['out_fusion'].argmax(1).cpu().numpy())
                    train_targets.extend(lbls.cpu().numpy())
                
                scheduler.step()
                
                # Validation
                model.eval()
                val_loss_sum = 0
                val_preds, val_targets = [], []
                
                with torch.no_grad():
                    for imgs, eegs, lbls in val_loader:
                        imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                        out = model(imgs, eegs)
                        loss = crit_cls(out['out_fusion'], lbls)
                        val_loss_sum += loss.item()
                        val_preds.extend(out['out_fusion'].argmax(1).cpu().numpy())
                        val_targets.extend(lbls.cpu().numpy())
                
                # 🔥 计算复杂指标
                val_acc = accuracy_score(val_targets, val_preds)
                val_f1 = f1_score(val_targets, val_preds, average='macro')
                train_acc = accuracy_score(train_targets, train_preds)
                
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    torch.save(model.state_dict(), best_model_path)
                    tqdm.write(f"💾 Saved Best (Ep{epoch+1}): Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")
                
                if val_f1 > best_f1_score:
                    best_f1_score = val_f1
                    torch.save(model.state_dict(), best_model_path_f1)
                    tqdm.write(f"💾 Saved Best F1 (Ep{epoch+1}): Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")

                pbar.set_postfix({
                    'T_Loss': f"{train_loss_sum/len(train_loader):.2f}",
                    'V_Loss': f"{val_loss_sum/len(val_loader):.2f}",
                    'T_Acc': f"{train_acc:.2f}",
                    'V_Acc': f"{val_acc:.2f}",
                    'V_F1': f"{val_f1:.2f}"
                })

            # --- Testing ---
            if os.path.exists(best_model_path_f1):
                model.load_state_dict(torch.load(best_model_path_f1, map_location=device))
            
            model.eval()
            test_preds, test_targets = [], []
            with torch.no_grad():
                for imgs, eegs, lbls in test_loader:
                    imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                    out = model(imgs, eegs)
                    test_preds.extend(out['out_fusion'].argmax(1).cpu().numpy())
                    test_targets.extend(lbls.cpu().numpy())
            
            print(f"\n👉 Subject {sub_id:02d} Test Results:")
            test_acc, test_f1 = print_metrics(test_targets, test_preds, class_names)
            test_accuracies.append(test_acc)

    print(f"\nOverall Mean Test Acc: {np.mean(test_accuracies):.4f}")

if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True, help="Path to faces_224 directory")
    parser.add_argument('-eeg_dir', type=str, required=True, help="Directory containing subXX/split.pkl")
    parser.add_argument('-vis_backbone', type=str, default='hsemotion', choices=['hsemotion', 'resnet'])
    parser.add_argument('-eeg_backbone', type=str, default='eegnet', choices=['eegnet', 'rgnn'])
    parser.add_argument('-alpha', type=float, default=0.1, help="Weight for CMD Loss")
    parser.add_argument('-beta', type=float, default=0.01, help="Weight for Diff Loss")
    
    args = parser.parse_args()
    main(args)