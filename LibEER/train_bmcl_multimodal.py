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
import argparse

# 引入 LibEER 工具
from config.setting import preset_setting, set_setting_by_args
from data_utils.split import get_split_index
from utils.args import get_args_parser
from utils.utils import setup_seed
from utils.store import make_output_dir
from data_utils.load_data import read_deap_preprocessed

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
                 common_dim=256, private_dim=256, eeg_length=7680, freeze_vis=True):
        super(CoupledModel, self).__init__()
        
        # --- Visual Branch ---
        if visual_backbone_name == 'hsemotion':
            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            self.vis_net = fer.model
            self.vis_feat_dim = 1280 
            if hasattr(self.vis_net, 'classifier'):
                if isinstance(self.vis_net.classifier, nn.Linear):
                    self.vis_feat_dim = self.vis_net.classifier.in_features
                self.vis_net.classifier = nn.Identity()
            elif hasattr(self.vis_net, 'fc'):
                if isinstance(self.vis_net.fc, nn.Linear):
                    self.vis_feat_dim = self.vis_net.fc.in_features
                self.vis_net.fc = nn.Identity()
        elif visual_backbone_name == 'resnet50':
            self.vis_net = models.resnet50(pretrained=True)
            self.vis_feat_dim = self.vis_net.fc.in_features
            self.vis_net.fc = nn.Identity()
            
        # 🔥 关键修改：冻结视觉骨干，防止过拟合和特征崩塌 🔥
        if freeze_vis:
            print("🔒 [Info] Visual Backbone is FREEZED to prevent overfitting.")
            for param in self.vis_net.parameters():
                param.requires_grad = False
        
        # --- EEG Branch ---
        if eeg_backbone_name == 'eegnet':
            self.eeg_net = EEGNet(num_electrodes=32, datapoints=eeg_length, num_classes=num_classes)
            self.eeg_feat_dim = self.eeg_net.fc.in_features
            self.eeg_net.fc = nn.Identity()
        elif eeg_backbone_name == 'rgnn':
            self.eeg_net = RGNN(num_electrodes=32, num_classes=num_classes, num_hidden=512, in_channels=eeg_length)
            self.eeg_feat_dim = 512

        # --- Projectors ---
        self.vis_common_fc = nn.Sequential(nn.Linear(self.vis_feat_dim, common_dim), nn.BatchNorm1d(common_dim), nn.ReLU())
        self.vis_private_fc = nn.Sequential(nn.Linear(self.vis_feat_dim, private_dim), nn.BatchNorm1d(private_dim), nn.ReLU())
        
        self.eeg_common_fc = nn.Sequential(nn.Linear(self.eeg_feat_dim, common_dim), nn.BatchNorm1d(common_dim), nn.ReLU())
        self.eeg_private_fc = nn.Sequential(nn.Linear(self.eeg_feat_dim, private_dim), nn.BatchNorm1d(private_dim), nn.ReLU())

        # --- Classifiers (Dropout increased) ---
        fusion_dim = common_dim + private_dim
        self.dropout = nn.Dropout(0.5)
        
        self.vis_classifier = nn.Linear(fusion_dim, num_classes)
        self.eeg_classifier = nn.Linear(fusion_dim, num_classes)
        self.fusion_final_classifier = nn.Linear(fusion_dim * 2, num_classes)

    def forward(self, img, eeg):
        if hasattr(self, 'vis_net'):
            vis_feat = self.vis_net(img)
        
        if isinstance(self.eeg_net, RGNN):
            edge_weight = torch.zeros((self.eeg_net.num_electrodes, self.eeg_net.num_electrodes), device=eeg.device)
            edge_weight[self.eeg_net.xs.to(eeg.device), self.eeg_net.ys.to(eeg.device)] = self.eeg_net.edge_weight
            edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
            x = F.relu(self.eeg_net.sgc(eeg, edge_weight))
            eeg_feat = self.eeg_net.pool(x)
        else:
            eeg_feat = self.eeg_net(eeg)

        vis_c = self.vis_common_fc(vis_feat)
        vis_p = self.vis_private_fc(vis_feat)
        eeg_c = self.eeg_common_fc(eeg_feat)
        eeg_p = self.eeg_private_fc(eeg_feat)

        vis_cp = torch.cat([vis_c, vis_p], dim=1)
        eeg_cp = torch.cat([eeg_c, eeg_p], dim=1)
        
        # Apply Dropout
        vis_cp_d = self.dropout(vis_cp)
        eeg_cp_d = self.dropout(eeg_cp)
        
        out_vis = self.vis_classifier(vis_cp_d)
        out_eeg = self.eeg_classifier(eeg_cp_d)
        
        fusion_feat = torch.cat([vis_cp, eeg_cp], dim=1)
        out_fusion = self.fusion_final_classifier(self.dropout(fusion_feat))

        return {'vis_c': vis_c, 'vis_p': vis_p, 'eeg_c': eeg_c, 'eeg_p': eeg_p,
                'out_vis': out_vis, 'out_eeg': out_eeg, 'out_fusion': out_fusion}

# ================= 3. 数据集与数据加载 =================

class MultimodalDataset(Dataset):
    def __init__(self, visual_paths, eeg_data_map, label_info, transform=None):
        self.visual_paths = visual_paths
        self.label_info = label_info 
        self.eeg_data_map = eeg_data_map
        self.transform = transform

    def __len__(self): return len(self.visual_paths)

    def __getitem__(self, idx):
        img_path = self.visual_paths[idx]
        sub_id, trial_id, label_val = self.label_info[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
        except: img = torch.zeros(3, 224, 224)
        
        eeg_tensor = torch.tensor(self.eeg_data_map[sub_id][trial_id], dtype=torch.float32)
        return img, eeg_tensor, torch.tensor(label_val, dtype=torch.long)

def get_aligned_data(args):
    print(f"Loading EEG Data from {args.dataset_path} ...")
    eeg_raw_data, _, eeg_raw_labels, _, _ = read_deap_preprocessed(args.dataset_path)
    eeg_map, eeg_lbl_map = {}, {}
    for sub_idx, (sub_dat, sub_lbl) in enumerate(zip(eeg_raw_data[0], eeg_raw_labels[0])):
        sub_id = sub_idx + 1
        eeg_map[sub_id] = {t_idx: dat for t_idx, dat in enumerate(sub_dat)}
        eeg_lbl_map[sub_id] = {t_idx: lbl for t_idx, lbl in enumerate(sub_lbl)}

    print("Aligning Visual Data...")
    struct_data, struct_label = [], []
    for sub_id in range(1, 33):
        sub_str = f"s{sub_id:02d}"
        sub_dir = os.path.join(args.faces_path, sub_str)
        sub_paths, sub_lbls = [], []
        
        if sub_id not in eeg_lbl_map:
            struct_data.append([]); struct_label.append([])
            continue

        for trial_id in range(1, 41):
            raw = eeg_lbl_map[sub_id][trial_id-1]
            cls = 0
            if raw[0] < 5 and raw[1] < 5: cls = 0
            elif raw[0] < 5 and raw[1] >= 5: cls = 1
            elif raw[0] >= 5 and raw[1] < 5: cls = 2
            elif raw[0] >= 5 and raw[1] >= 5: cls = 3
            
            pattern = f"{sub_str}_trial{trial_id:02d}_seg*.jpg"
            files = sorted(glob.glob(os.path.join(sub_dir, pattern)), key=lambda x: int(x.split('_seg')[-1].split('.')[0]))
            
            if files:
                sub_paths.append(files)
                sub_lbls.append([cls] * len(files))
            else:
                sub_paths.append([])
                sub_lbls.append([])
        
        struct_data.append(sub_paths)
        struct_label.append(sub_lbls)
    return struct_data, struct_label, eeg_map

def flatten(data_trails, label_trails, indices, sub_id):
    flat_paths, flat_meta = [], []
    for idx in indices:
        if idx < len(data_trails):
            paths = data_trails[idx]
            lbls = label_trails[idx]
            if paths:
                flat_paths.extend(paths)
                for lbl in lbls: flat_meta.append((sub_id, idx, lbl))
    return flat_paths, flat_meta

# ================= 4. Main =================

def main(args):
    if not args.output_dir: args.output_dir = make_output_dir(args, "BMCL_Result")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 🔥 安全获取参数，防止 AttributeError
    alpha_val = getattr(args, 'alpha', 0.1)
    beta_val = getattr(args, 'beta', 0.01)
    
    struct_data, struct_label, eeg_map = get_aligned_data(args)
    
    sample_sub = list(eeg_map.keys())[0]
    sample_trial = list(eeg_map[sample_sub].keys())[0]
    eeg_length_detected = eeg_map[sample_sub][sample_trial].shape[1]
    print(f"✅ Detected EEG Length: {eeg_length_detected}")

    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_val = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    subjects = list(enumerate(zip(struct_data, struct_label), 1))
    if args.subjects_limit > 0: subjects = subjects[:args.subjects_limit]
    
    test_accs = []
    
    for rridx, (d_trails, l_trails) in subjects:
        if not d_trails: continue
        
        sub_split_path = os.path.join(args.eeg_dir, f"sub{rridx:02d}", "split.pkl")
        if os.path.exists(sub_split_path):
            with open(sub_split_path, 'rb') as f: tts = pickle.load(f)
            print(f"Loaded split for S{rridx}")
        else:
            tts = get_split_index(d_trails, l_trails, set_setting_by_args(args))

        for fold_idx, (tr_idx, te_idx, val_idx) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            if val_idx[0] == -1: val_idx = te_idx
            
            tr_p, tr_m = flatten(d_trails, l_trails, tr_idx, rridx)
            va_p, va_m = flatten(d_trails, l_trails, val_idx, rridx)
            te_p, te_m = flatten(d_trails, l_trails, te_idx, rridx)
            
            # 使用大 Batch Size (因为冻结了骨干)
            train_loader = DataLoader(MultimodalDataset(tr_p, eeg_map, tr_m, transform=transforms_train), 
                                      batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(MultimodalDataset(va_p, eeg_map, va_m, transform=transforms_val), 
                                    batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(MultimodalDataset(te_p, eeg_map, te_m, transform=transforms_val), 
                                     batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            # 默认冻结视觉骨干
            freeze_vis = not getattr(args, 'unfreeze_vis', False)
            model = CoupledModel(getattr(args, 'vis_backbone', 'hsemotion'), 
                                 getattr(args, 'eeg_backbone', 'eegnet'), 
                                 eeg_length=eeg_length_detected, 
                                 freeze_vis=freeze_vis).to(device)
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            crit_cls = nn.CrossEntropyLoss()
            crit_cmd = CMDLoss()
            crit_diff = DiffLoss()
            
            best_acc = 0.0
            best_state = None
            
            pbar = tqdm(range(args.epochs), desc=f"S{rridx}-Fold{fold_idx}", leave=False)
            for ep in pbar:
                model.train()
                for imgs, eegs, lbls in train_loader:
                    imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                    optimizer.zero_grad()
                    out = model(imgs, eegs)
                    
                    l_task = crit_cls(out['out_vis'], lbls) + crit_cls(out['out_eeg'], lbls) + crit_cls(out['out_fusion'], lbls)
                    l_sim = alpha_val * crit_cmd(out['vis_c'], out['eeg_c'])
                    l_diff = beta_val * (crit_diff(out['vis_c'], out['vis_p']) + crit_diff(out['eeg_c'], out['eeg_p']))
                    
                    loss = l_task + l_sim + l_diff
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                model.eval()
                c_vis, c_eeg, c_fus, tot = 0, 0, 0, 0
                with torch.no_grad():
                    for imgs, eegs, lbls in val_loader:
                        imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                        out = model(imgs, eegs)
                        c_vis += (out['out_vis'].argmax(1) == lbls).sum().item()
                        c_eeg += (out['out_eeg'].argmax(1) == lbls).sum().item()
                        c_fus += (out['out_fusion'].argmax(1) == lbls).sum().item()
                        tot += lbls.size(0)
                
                acc_fus = c_fus / tot
                acc_vis = c_vis / tot
                acc_eeg = c_eeg / tot
                
                if acc_fus > best_acc:
                    best_acc = acc_fus
                    best_state = model.state_dict()
                
                pbar.set_postfix({'Val': f"{acc_fus:.3f}", 'V': f"{acc_vis:.2f}", 'E': f"{acc_eeg:.2f}"})
            
            if best_state: model.load_state_dict(best_state)
            model.eval()
            c_vis, c_eeg, c_fus, tot = 0, 0, 0, 0
            with torch.no_grad():
                for imgs, eegs, lbls in test_loader:
                    imgs, eegs, lbls = imgs.to(device), eegs.to(device), lbls.to(device)
                    out = model(imgs, eegs)
                    c_vis += (out['out_vis'].argmax(1) == lbls).sum().item()
                    c_eeg += (out['out_eeg'].argmax(1) == lbls).sum().item()
                    c_fus += (out['out_fusion'].argmax(1) == lbls).sum().item()
                    tot += lbls.size(0)
            
            print(f"Sub{rridx} Test Result | Vis: {c_vis/tot:.4f} | EEG: {c_eeg/tot:.4f} | Fusion: {c_fus/tot:.4f}")
            test_accs.append(c_fus/tot)

    print(f"Overall Mean Test Acc: {np.mean(test_accs):.4f}")

if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True)
    #parser.add_argument('-dataset_path', type=str, required=True)
    parser.add_argument('-eeg_dir', type=str, required=True)
    # 🔥 改为双横线，避免冲突
    parser.add_argument('--vis_backbone', type=str, default='hsemotion')
    parser.add_argument('--eeg_backbone', type=str, default='eegnet')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('-subjects_limit', type=int, default=0)
    parser.add_argument('--unfreeze_vis', action='store_true', help="Unfreeze visual backbone (not recommended)")
    
    main(parser.parse_args())