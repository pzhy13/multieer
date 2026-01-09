import os
import glob
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 🔴 HSEmotion 补丁
_original_load = torch.load
def safe_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_patch

from utils.args import get_args_parser
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, get_split_index, index_to_data
from utils.utils import setup_seed
from models.Models import Model as ModelDict

# ================= OGM-GE 优化版 =================
class OGM_GE_Optimized(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        
    def __call__(self, score_eeg, score_vis, target, epoch_ratio):
        p_eeg = torch.softmax(score_eeg, dim=1)
        p_vis = torch.softmax(score_vis, dim=1)
        
        idx = target.view(-1, 1)
        conf_eeg = p_eeg.gather(1, idx).squeeze()
        conf_vis = p_vis.gather(1, idx).squeeze()

        score_eeg_sum = conf_eeg.sum()
        score_vis_sum = conf_vis.sum()
        
        if score_vis_sum == 0: score_vis_sum = 1e-8
        if score_eeg_sum == 0: score_eeg_sum = 1e-8

        ratio = score_eeg_sum / score_vis_sum
        
        coef_eeg = 1.0
        coef_vis = 1.0
        
        if ratio > 1:
            coef_eeg = 1.0 - torch.tanh(self.alpha * ratio)
        else:
            coef_vis = 1.0 - torch.tanh(self.alpha * (1.0 / ratio))
            
        self.register_hook(score_eeg, coef_eeg, epoch_ratio)
        self.register_hook(score_vis, coef_vis, epoch_ratio)
        
        return score_eeg, score_vis

    def register_hook(self, tensor, coef, epoch_ratio):
        if not isinstance(coef, torch.Tensor):
            coef = torch.tensor(coef).to(tensor.device)
            
        def hook_fn(grad):
            new_grad = grad * coef
            if epoch_ratio > 0:
                std = grad.std().item() if grad.numel() > 1 else 1e-4
                noise = torch.zeros_like(grad).normal_(0, std + 1e-8)
                new_grad = new_grad + noise
            return new_grad
            
        tensor.register_hook(hook_fn)
# ===================================================

class PairedDataset(Dataset):
    def __init__(self, eeg_trials, face_trials, label_trials, trial_indices, transform=None):
        self.samples = []
        self.transform = transform
        for t in trial_indices:
            if t < 0 or t >= len(eeg_trials): continue
            e_trial = eeg_trials[t]
            f_trial = face_trials[t] if t < len(face_trials) else []
            lab_trial = label_trials[t] if t < len(label_trials) else None
            if e_trial is None: continue
            
            e_trial = np.array(e_trial)
            num_segments = len(e_trial)
            for s in range(num_segments):
                eeg_seg = e_trial[s]
                img_path = f_trial[s] if s < len(f_trial) else None
                lbl = 0
                if lab_trial is not None:
                    try:
                        l = lab_trial[s]
                        if hasattr(l, '__len__') and not isinstance(l, (int, np.integer)):
                            lbl = int(np.argmax(l))
                        else:
                            lbl = int(l)
                    except: lbl = 0
                self.samples.append((eeg_seg.astype('float32'), img_path, lbl))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        eeg_seg, img_path, lbl = self.samples[idx]
        eeg_tensor = torch.tensor(eeg_seg)
        
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
            except: img = torch.zeros(3,224,224)
        else: img = torch.zeros(3,224,224)
        
        return eeg_tensor, img, int(lbl)

def build_face_index(dataset_path, faces_path):
    all_labels = {}
    for sub_id in range(1, 33):
        p = os.path.join(dataset_path, f's{sub_id:02d}.dat')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                all_labels[sub_id] = content['labels']
    face_index = []
    for sub_id in range(1, 33):
        sub_str = f's{sub_id:02d}'
        sub_face_dir = os.path.join(faces_path, sub_str)
        trials = []
        if not os.path.exists(sub_face_dir) or sub_id not in all_labels:
            face_index.append(trials)
            continue
        for trial_id in range(1, 41):
            pattern = f"{sub_str}_trial{trial_id:02d}_seg*.jpg"
            files = glob.glob(os.path.join(sub_face_dir, pattern))
            files.sort(key=lambda x: int(x.split('_seg')[-1].split('.')[0]) if '_seg' in x else 0)
            trials.append(files)
        face_index.append(trials)
    return face_index

def instantiate_visual_model(name, device, num_classes):
    import torchvision.models as models
    
    # 分类头：BatchNorm + Dropout + Linear
    def create_classifier(in_features, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.7), # 标准 Dropout
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.7),
            nn.Linear(256, num_classes)
        )

    model = None
    if name == 'hsemotion':
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            model = fer.model
            num_ftrs = 1280
            try:
                if hasattr(model, 'num_features'): num_ftrs = model.num_features
                elif hasattr(model, 'classifier'): num_ftrs = model.classifier.in_features
            except: pass
            
            # 🔓 不再冻结，允许微调 (requires_grad=True by default)
            
            model.classifier = create_classifier(num_ftrs, num_classes)
            if hasattr(model, 'fc'): model.fc = nn.Identity()
            print("INFO: HSEmotion loaded for FINE-TUNING.")
        except Exception as e: 
            print(f"Warning: HSEmotion load failed ({e}), fallback to ResNet18")
    
    if model is None:
        model = models.resnet18(pretrained=True)
        model.fc = create_classifier(model.fc.in_features, num_classes)
        print("INFO: ResNet18 loaded for FINE-TUNING.")
        
    return model.to(device)

def instantiate_eeg_model(name, channels, feature_dim, num_classes, device):
    if name == 'EEGNet':
        from models.EEGNet import EEGNet
        return EEGNet(num_electrodes=channels, datapoints=feature_dim, num_classes=num_classes).to(device)
    
    if name in ModelDict:
        try:
            return ModelDict[name]().to(device)
        except:
            return ModelDict[name](input_shape=(1, channels, feature_dim), num_classes=num_classes).to(device)

    class SimpleEEG(nn.Module):
        def __init__(self, c, L, nc):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c, 32, 3, padding=1), 
                nn.BatchNorm1d(32),
                nn.ReLU(), 
                nn.Flatten(), 
                #nn.Dropout(p=0.5), 
                nn.Linear(32*L, nc)
            )
        def forward(self,x): return self.net(x)
    return SimpleEEG(channels, feature_dim, num_classes).to(device)

def main():
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True)
    parser.add_argument('-eeg_model', type=str, default='EEGNet')
    parser.add_argument('-vis_model', type=str, default='hsemotion') 
    parser.add_argument('-fusion', type=str, default='late', choices=['late','concat'])
    args = parser.parse_args()
    
    if args.label_used is None: args.label_used = ['valence', 'arousal']
    if args.bounds is None: args.bounds = [5.0, 5.0]
    if not args.onehot: args.onehot = True

    if args.setting is None:
        setting = set_setting_by_args(args)
    else:
        setting = preset_setting[args.setting](args)

    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    face_index = build_face_index(args.dataset_path, args.faces_path)

    # ✂️ 适度增强：移除高斯噪声，保留随机裁剪和翻转
    train_transform = transforms.Compose([
        # 将最小比例从 0.85 调低到 0.5 或 0.4，强迫模型通过局部特征（如眼睛、嘴巴）识别情绪
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
        transforms.RandomHorizontalFlip(),
        # 增加颜色抖动的强度
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2), # 20% 概率变黑白，防止过分依赖颜色
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eeg_model = instantiate_eeg_model(args.eeg_model, channels, feature_dim, num_classes, device)
    vis_model = instantiate_visual_model(args.vis_model, device, num_classes)
    
    mix_param = nn.Parameter(torch.tensor(0.0, device=device))
    ogm_ge = OGM_GE_Optimized(alpha=0.5)

    # 🚀 差分学习率策略 (Differential Learning Rates)
    vis_backbone_params = []
    vis_head_params = []
    for name, param in vis_model.named_parameters():
        if 'classifier' in name or 'fc' in name:
            vis_head_params.append(param)
        else:
            vis_backbone_params.append(param)

    params = [
        {'params': eeg_model.parameters(), 'lr': 0.01},          # EEG: 正常高LR
        {'params': vis_backbone_params, 'lr': 1e-5},             # 视觉骨干: 极低LR (微调)
        {'params': vis_head_params, 'lr': 1e-3},                 # 视觉头: 正常LR
        {'params': [mix_param], 'lr': args.lr}
    ]
    
    optimizer = optim.Adam(params, weight_decay=1e-2) # 恢复标准 weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.25) # 降低一点平滑力度

    for sub_idx, (data_sub, label_sub) in enumerate(zip(data, label), 1):
        if len(data_sub) == 0: continue
        print(f"\nSubject {sub_idx}...")
        tts = get_split_index(data_sub, label_sub, setting)
        
        for ridx, (train_idx, test_idx, val_idx) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            
            tr_data, tr_label, val_data, val_label, te_data, te_label = index_to_data(
                data_sub, label_sub, train_idx, test_idx, val_idx, keep_dim=True
            )
            
            face_trials_all = face_index[sub_idx-1] if sub_idx-1 < len(face_index) else []
            if not any(len(t)>0 for t in face_trials_all): continue

            face_train = [face_trials_all[i] for i in train_idx]
            face_val = [face_trials_all[i] for i in val_idx] if val_idx[0] != -1 else []
            face_test = [face_trials_all[i] for i in test_idx]

            ds_train = PairedDataset(tr_data, face_train, tr_label, list(range(len(tr_data))), transform=train_transform)
            ds_val = PairedDataset(val_data, face_val, val_label, list(range(len(val_data))), transform=val_test_transform)
            ds_test = PairedDataset(te_data, face_test, te_label, list(range(len(te_data))), transform=val_test_transform)

            loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
            loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
            loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

            best_val = 0.0
            best_model_path = os.path.join('checkpoints', f'fusion_sub{sub_idx}_fold{ridx}_best.pth')
            
            for epoch in range(args.epochs):
                eeg_model.train(); vis_model.train()
                total, correct = 0, 0
                
                loop = tqdm(loader_train, desc=f"S{sub_idx} F{ridx} Ep{epoch+1}", leave=False)
                
                for batch in loop:
                    eegs, imgs, targets = batch
                    eegs, imgs, targets = eegs.to(device), imgs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    out_eeg = eeg_model(eegs)
                    out_vis = vis_model(imgs)
                    
                    score_eeg, score_vis = ogm_ge(out_eeg, out_vis, targets, epoch / args.epochs)
                    
                    alpha = torch.sigmoid(mix_param)
                    logits = alpha * score_eeg + (1 - alpha) * score_vis

                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()

                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    
                    loop.set_postfix(loss=loss.item())

                scheduler.step()
                
                train_acc = correct / total if total > 0 else 0
                
                eeg_model.eval(); vis_model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for batch in loader_val:
                        eegs, imgs, targets = batch
                        eegs, imgs, targets = eegs.to(device), imgs.to(device), targets.to(device)
                        
                        out_e = eeg_model(eegs)
                        out_v = vis_model(imgs)
                        
                        alpha = torch.sigmoid(mix_param)
                        logits = alpha * out_e + (1 - alpha) * out_v
                        
                        val_correct += (logits.argmax(dim=1) == targets).sum().item()
                        val_total += targets.size(0)
                
                val_acc = val_correct / val_total if val_total > 0 else 0
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                     print(f"Sub {sub_idx} Fold {ridx} Ep {epoch+1}: Train {train_acc:.4f} Val {val_acc:.4f}")
                
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save({
                        'eeg_state': eeg_model.state_dict(),
                        'vis_state': vis_model.state_dict(),
                        'mix_param': mix_param, 
                    }, best_model_path)

            print(f"Training finished. Loading best model (Val Acc: {best_val:.4f})...")
            
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                eeg_model.load_state_dict(checkpoint['eeg_state'])
                vis_model.load_state_dict(checkpoint['vis_state'])
                with torch.no_grad():
                    mix_param.copy_(checkpoint['mix_param'])
            else:
                print("Warning: No best model found. Using last epoch model.")

            eeg_model.eval(); vis_model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for batch in tqdm(loader_test, desc=f"Testing S{sub_idx} F{ridx}"):
                    eegs, imgs, targets = batch
                    eegs, imgs, targets = eegs.to(device), imgs.to(device), targets.to(device)
                    
                    out_e = eeg_model(eegs)
                    out_v = vis_model(imgs)
                    
                    alpha = torch.sigmoid(mix_param)
                    logits = alpha * out_e + (1 - alpha) * out_v
                    
                    test_correct += (logits.argmax(dim=1) == targets).sum().item()
                    test_total += targets.size(0)
            
            test_acc = test_correct / test_total if test_total > 0 else 0
            print(f"👉 Subject {sub_idx} Fold {ridx} TEST ACCURACY: {test_acc:.4f}\n")

if __name__ == '__main__':
    main()