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
from sklearn.metrics import f1_score, accuracy_score

# 引入 LibEER 工具
from config.setting import preset_setting, set_setting_by_args
from data_utils.split import get_split_index, merge_to_part
from utils.args import get_args_parser
from utils.utils import setup_seed
from utils.store import make_output_dir
from data_utils.load_data import get_data

# 引入模型
from models.EEGNet import EEGNet
from models.RGNN import RGNN

# 🔴 HSEmotion 补丁
_original_load = torch.load
def safe_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_patch
from hsemotion.facial_emotions import HSEmotionRecognizer

# ================= 1. 损失函数 =================

class CMDLoss(nn.Module):
    def __init__(self, k=5):
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
                 common_dim=128, private_dim=128, feature_dim=128, eeg_channels=32, freeze_vis=True):
        super(CoupledModel, self).__init__()
        
        # --- Visual Branch ---
        if visual_backbone_name == 'hsemotion':
            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            self.vis_net = fer.model
            self.vis_feat_dim = 1280 
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
            # feature_dim = 128 (Time Length)
            self.eeg_net = EEGNet(num_electrodes=eeg_channels, datapoints=feature_dim, num_classes=num_classes)
            self.eeg_feat_dim = self.eeg_net.fc.in_features
            self.eeg_net.fc = nn.Identity()
        elif eeg_backbone_name == 'rgnn':
            self.eeg_net = RGNN(num_electrodes=eeg_channels, num_classes=num_classes, num_hidden=512, in_channels=feature_dim)
            self.eeg_feat_dim = 512

        # --- Projectors ---
        self.vis_common_fc = nn.Sequential(nn.Linear(self.vis_feat_dim, common_dim), nn.BatchNorm1d(common_dim), nn.ReLU())
        self.vis_private_fc = nn.Sequential(nn.Linear(self.vis_feat_dim, private_dim), nn.BatchNorm1d(private_dim), nn.ReLU())
        
        self.eeg_common_fc = nn.Sequential(nn.Linear(self.eeg_feat_dim, common_dim), nn.BatchNorm1d(common_dim), nn.ReLU())
        self.eeg_private_fc = nn.Sequential(nn.Linear(self.eeg_feat_dim, private_dim), nn.BatchNorm1d(private_dim), nn.ReLU())

        # --- Classifiers ---
        fusion_dim = common_dim + private_dim
        self.dropout = nn.Dropout(0.5)
        
        self.vis_classifier = nn.Linear(fusion_dim, num_classes)
        self.eeg_classifier = nn.Linear(fusion_dim, num_classes)
        self.fusion_final_classifier = nn.Linear(fusion_dim * 2, num_classes)

    def forward(self, img, eeg):
        if hasattr(self, 'vis_net'):
            vis_feat = self.vis_net(img)
        
        if self.eeg_backbone_name == 'rgnn':
            if eeg.dim() == 4 and eeg.size(1) == 1: eeg = eeg.squeeze(1)
            eeg_feat = self.eeg_net(eeg)
        else:
            eeg_feat = self.eeg_net(eeg)

        vis_c = self.vis_common_fc(vis_feat)
        vis_p = self.vis_private_fc(vis_feat)
        eeg_c = self.eeg_common_fc(eeg_feat)
        eeg_p = self.eeg_private_fc(eeg_feat)

        vis_cp = torch.cat([vis_c, vis_p], dim=1)
        eeg_cp = torch.cat([eeg_c, eeg_p], dim=1)
        
        vis_cp_d = self.dropout(vis_cp)
        eeg_cp_d = self.dropout(eeg_cp)
        
        out_vis = self.vis_classifier(vis_cp_d)
        out_eeg = self.eeg_classifier(eeg_cp_d)
        
        fusion_feat = torch.cat([vis_cp, eeg_cp], dim=1)
        out_fusion = self.fusion_final_classifier(self.dropout(fusion_feat))

        return {'vis_c': vis_c, 'vis_p': vis_p, 'eeg_c': eeg_c, 'eeg_p': eeg_p,
                'out_vis': out_vis, 'out_eeg': out_eeg, 'out_fusion': out_fusion}

# ================= 3. 数据集与对齐 =================

class MultimodalDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list 
        self.transform = transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        img_path, eeg_data, label_val = self.data_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
        except: 
            img = torch.zeros(3, 224, 224)
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        return img, eeg_tensor, torch.tensor(label_val, dtype=torch.long)

def get_visual_paths_map(args):
    """ (SubID, TrialID, SegID) -> ImagePath """
    print(f"Indexing Visual Data from {args.faces_path}...")
    visual_map = {}
    for sub_id in range(1, 33):
        sub_str = f"s{sub_id:02d}"
        sub_dir = os.path.join(args.faces_path, sub_str)
        if not os.path.exists(sub_dir): continue
        search_path = os.path.join(sub_dir, "*.jpg")
        files = glob.glob(search_path)
        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                parts = fname.split('_')
                t_id = int(parts[1].replace('trial', ''))
                s_id = int(parts[2].replace('seg', '').split('.')[0])
                visual_map[(sub_id, t_id, s_id)] = fpath
            except: continue
    print(f"Indexed {len(visual_map)} images.")
    return visual_map

def prepare_split_data(eeg_data, eeg_label, visual_map, trial_indices, sub_id):
    """ 
    trial_indices: List of Trial IDs (0-39)
    eeg_data: List of Trials, e.g., (40, 60, 32, 128)
    我们需要将选中的 Trial 展开成 Samples。
    """
    dataset_list = []
    
    for t_idx in trial_indices:
        if t_idx >= len(eeg_data): continue
        
        # 获取整个 Trial 的数据 (NumSamples, Channels, Time) -> (60, 32, 128)
        trial_samples = eeg_data[t_idx]
        trial_labels = eeg_label[t_idx] # (60, 4) or (60,)
        
        num_samples_in_trial = len(trial_samples)
        
        for s_idx in range(num_samples_in_trial):
            # 获取对应的 Image Path
            # Trial ID 是 1-based (t_idx + 1)
            # Sample ID (Seg ID) 是 0-based (s_idx)
            img_path = visual_map.get((sub_id, t_idx + 1, s_idx), None)
            
            if img_path:
                # 提取单个切片 (Channels, Time) -> (32, 128)
                eeg_sample = trial_samples[s_idx]
                label_sample = trial_labels[s_idx]
                
                if isinstance(label_sample, np.ndarray) and label_sample.ndim > 0:
                    label_val = np.argmax(label_sample)
                else:
                    label_val = int(label_sample)
                
                dataset_list.append((img_path, eeg_sample, label_val))
            
    return dataset_list

# ================= 4. 主程序 =================

def main(args):
    if args.output_dir is None:
        args.output_dir = make_output_dir(args, "BMCL_Result")
    else:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    if args.setting is None: setting = set_setting_by_args(args)
    else: setting = preset_setting[args.setting](args)

    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("Loading EEG Data (Time Series Segmented)...")
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data_all, label_all = merge_to_part(data, label, setting)
    
    print(f"Data Loaded. Subjects: {len(data_all)}, Time Length: {feature_dim}, Classes: {num_classes}")

    visual_map = get_visual_paths_map(args)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for sub_idx, (sub_eeg, sub_label) in enumerate(zip(data_all, label_all)):
        sub_id = sub_idx + 1 
        print(f"\n========== Processing Subject {sub_id:02d} ==========")
        
        split_file = os.path.join(args.eeg_dir, f"sub{sub_id:02d}", "split.pkl")
        
        if os.path.exists(split_file):
            print(f"Loading split from: {split_file}")
            with open(split_file, 'rb') as f:
                tts = pickle.load(f)
        else:
            print(f"Warning: Split file not found at {split_file}. Using dynamic split.")
            setup_seed(args.seed + sub_id)
            tts = get_split_index(sub_eeg, sub_label, setting)

        for fold_idx, (train_idx, test_idx, val_idx) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            if val_idx[0] == -1 or len(val_idx) == 0: val_idx = test_idx
            
            # 这里传入的是 Trial Indices，函数内部会展开为 Samples
            train_list = prepare_split_data(sub_eeg, sub_label, visual_map, train_idx, sub_id)
            val_list = prepare_split_data(sub_eeg, sub_label, visual_map, val_idx, sub_id)
            test_list = prepare_split_data(sub_eeg, sub_label, visual_map, test_idx, sub_id)
            
            if len(train_list) == 0:
                print(f"Skipping S{sub_id} Fold {fold_idx} (No data)")
                continue

            print(f"S{sub_id} Fold {fold_idx}: Train Samples: {len(train_list)}, Val Samples: {len(val_list)}")

            train_loader = DataLoader(MultimodalDataset(train_list, transform=train_transform), 
                                      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(MultimodalDataset(val_list, transform=val_transform), 
                                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(MultimodalDataset(test_list, transform=val_transform), 
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            model = CoupledModel(
                visual_backbone_name=args.vis_backbone,
                eeg_backbone_name=args.eeg_backbone,
                num_classes=num_classes,
                feature_dim=feature_dim,
                eeg_channels=channels,
                freeze_vis=True
            ).to(device)
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=args.lr, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            crit_cls = nn.CrossEntropyLoss()
            crit_cmd = CMDLoss()
            crit_diff = DiffLoss()
            
            best_f1 = 0.0
            best_path = os.path.join(args.output_dir, f"sub{sub_id:02d}", f"checkpoint-bestmacro-f1.pth")
            if not os.path.exists(os.path.dirname(best_path)): os.makedirs(os.path.dirname(best_path))

            pbar = tqdm(range(args.epochs), desc=f"S{sub_id}-Fold{fold_idx}", leave=False)
            for epoch in pbar:
                model.train()
                train_loss = 0
                for imgs, eegs, lbls in train_loader:
                    imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                    optimizer.zero_grad()
                    out = model(imgs, eegs)
                    
                    l_task = crit_cls(out['out_vis'], lbls) + crit_cls(out['out_eeg'], lbls) + crit_cls(out['out_fusion'], lbls)
                    l_sim = args.alpha * crit_cmd(out['vis_c'], out['eeg_c'])
                    l_diff = args.beta * (crit_diff(out['vis_c'], out['vis_p']) + crit_diff(out['eeg_c'], out['eeg_p']))
                    
                    loss = l_task + l_sim + l_diff
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for imgs, eegs, lbls in val_loader:
                        imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                        out = model(imgs, eegs)
                        preds = out['out_fusion'].argmax(1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(lbls.cpu().numpy())
                
                val_acc = accuracy_score(all_labels, all_preds)
                val_f1 = f1_score(all_labels, all_preds, average='macro')
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save(model.state_dict(), best_path)
                
                pbar.set_postfix({'Loss': f"{train_loss/len(train_loader):.2f}", 'Val_F1': f"{val_f1:.4f}", 'Val_Acc': f"{val_acc:.4f}"})

            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
            
            model.eval()
            test_preds, test_labels = [], []
            with torch.no_grad():
                for imgs, eegs, lbls in test_loader:
                    imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                    out = model(imgs, eegs)
                    test_preds.extend(out['out_fusion'].argmax(1).cpu().numpy())
                    test_labels.extend(lbls.cpu().numpy())
            
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            
            print(f"Subject {sub_id:02d} Result | Val Best F1: {best_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            print(f"Model saved to: {best_path}")

if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True)
    parser.add_argument('-eeg_dir', type=str, required=True, help="Path containing the split.pkl files")
    parser.add_argument('-vis_backbone', type=str, default='hsemotion', choices=['hsemotion', 'resnet'])
    parser.add_argument('-eeg_backbone', type=str, default='eegnet', choices=['eegnet', 'rgnn'])
    parser.add_argument('-alpha', type=float, default=0.1, help="Weight for CMD Loss")
    parser.add_argument('-beta', type=float, default=0.01, help="Weight for Diff Loss")
    
    args = parser.parse_args()
    main(args)