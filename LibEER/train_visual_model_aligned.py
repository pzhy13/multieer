import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score  # 🔴 新增

# 引入 LibEER 工具
from config.setting import preset_setting, set_setting_by_args
from data_utils.split import get_split_index
from utils.args import get_args_parser
from utils.utils import setup_seed
from utils.store import make_output_dir

# 🔴 HSEmotion 补丁
_original_load = torch.load
def safe_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_patch
from hsemotion.facial_emotions import HSEmotionRecognizer

# ================= 早停类 =================
class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ================= 数据集类 =================
class VisualDataset(Dataset):
    def __init__(self, data_list, label_list, transform=None):
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        label_val = self.label_list[idx] 
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
        except: img = torch.zeros(3, 224, 224)
        return img, torch.tensor(label_val, dtype=torch.long)

def get_visual_data_aligned(args):
    print(f"Loading Visual Data Index...")
    all_labels = {}
    for sub_id in range(1, 33):
        path = os.path.join(args.dataset_path, f"s{sub_id:02d}.dat")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                all_labels[sub_id] = content['labels']

    aligned_data = [] 
    aligned_label = [] 

    for sub_id in range(1, 33):
        sub_str = f"s{sub_id:02d}"
        sub_face_dir = os.path.join(args.faces_path, sub_str)
        
        if not os.path.exists(sub_face_dir) or sub_id not in all_labels:
            aligned_data.append([])
            aligned_label.append([])
            continue

        sub_trails_data = []
        sub_trails_label = []

        for trial_id in range(1, 41):
            if trial_id > len(all_labels[sub_id]): break
            raw_label = all_labels[sub_id][trial_id - 1]
            valence, arousal = raw_label[0], raw_label[1]
            v_high = valence >= 5
            a_high = arousal >= 5
            
            if not v_high and not a_high: cls = 0   
            elif not v_high and a_high:   cls = 1   
            elif v_high and not a_high:   cls = 2   
            elif v_high and a_high:       cls = 3   
            
            pattern = f"{sub_str}_trial{trial_id:02d}_seg*.jpg"
            search_path = os.path.join(sub_face_dir, pattern)
            files = glob.glob(search_path)
            files.sort(key=lambda x: int(x.split('_seg')[-1].split('.')[0]))
            
            if len(files) > 0:
                sub_trails_data.append(files)
                sub_trails_label.append([cls] * len(files))
            else:
                sub_trails_data.append([])
                sub_trails_label.append([])

        aligned_data.append(sub_trails_data)
        aligned_label.append(sub_trails_label)
    return aligned_data, aligned_label

def flatten_data(data_trails, label_trails, indices):
    flat_data = []
    flat_label = []
    for i in indices:
        if i < len(data_trails):
            flat_data.extend(data_trails[i])
            flat_label.extend(label_trails[i])
    return flat_data, flat_label

# ================= 主程序 =================
def main(args):
    if not hasattr(args, 'output_dir') or args.output_dir is None:
        args.output_dir = make_output_dir(args, "VisualModel")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.setting is None: setting = set_setting_by_args(args)
    else: setting = preset_setting[args.setting](args)

    setup_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_all_subs, label_all_subs = get_visual_data_aligned(args)
    
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
    test_f1s = []  # 🔴 新增

    target_subjects = list(enumerate(zip(data_all_subs, label_all_subs), 1))
    if hasattr(args, 'subjects_limit') and args.subjects_limit > 0:
        target_subjects = target_subjects[:args.subjects_limit]

    for rridx, (data_trails, label_trails) in target_subjects:
        if len(data_trails) == 0: continue
        
        eeg_sub_dir = os.path.join(args.eeg_dir, f"sub{rridx:02d}")
        split_path = os.path.join(eeg_sub_dir, 'split.pkl')
        
        if os.path.exists(split_path):
            with open(split_path, 'rb') as f: tts = pickle.load(f)
        else:
            setup_seed(args.seed + rridx)
            tts = get_split_index(data_trails, label_trails, setting)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            if val_indexes[0] == -1 or len(val_indexes) == 0: val_indexes = test_indexes

            train_paths, train_lbls = flatten_data(data_trails, label_trails, train_indexes)
            val_paths, val_lbls = flatten_data(data_trails, label_trails, val_indexes)
            test_paths, test_lbls = flatten_data(data_trails, label_trails, test_indexes)

            train_set = VisualDataset(train_paths, train_lbls, transform=train_transform)
            val_set = VisualDataset(val_paths, val_lbls, transform=val_test_transform)
            test_set = VisualDataset(test_paths, test_lbls, transform=val_test_transform)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)

            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            model = fer.model
            num_ftrs = 1280
            try:
                if hasattr(model, 'classifier') and not isinstance(model.classifier, nn.Identity): num_ftrs = model.classifier.in_features
                elif hasattr(model, 'fc') and not isinstance(model.fc, nn.Identity): num_ftrs = model.fc.in_features
            except: pass
            
            if not args.unfreeze_backbone:
                for param in model.parameters(): param.requires_grad = False
            
            model.classifier = nn.Sequential(
                nn.Dropout(p=getattr(args, 'dropout', 0.5)),
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(p=getattr(args, 'dropout', 0.5)),
                nn.Linear(256, 4)
            )
            if hasattr(model, 'fc'): model.fc = nn.Identity()
            model.to(device)

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-2))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.1))

            best_model_path = os.path.join(args.output_dir, f"visual_model_sub{rridx}_fold{ridx}_bestf1.pth")
            best_val_f1 = 0.0  # 🔴 关键修改

            pbar = tqdm(range(args.epochs), desc=f"S{rridx}", leave=False)
            for epoch in pbar:
                model.train()
                for imgs, targets in train_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                val_preds, val_gts = [], []
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs = imgs.to(device)
                        outputs = model(imgs)
                        val_preds.extend(outputs.argmax(1).cpu().numpy())
                        val_gts.extend(targets.numpy())

                val_f1 = f1_score(val_gts, val_preds, average='macro')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), best_model_path)

                pbar.set_postfix({'V_F1': f"{val_f1:.3f}"})

            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            test_preds, test_gts = [], []
            with torch.no_grad():
                for imgs, targets in test_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    test_preds.extend(outputs.argmax(1).cpu().numpy())
                    test_gts.extend(targets.numpy())

            test_acc = np.mean(np.array(test_preds) == np.array(test_gts))
            test_f1 = f1_score(test_gts, test_preds, average='macro')

            print(f"👉 Subject {rridx} Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            test_accuracies.append(test_acc)
            test_f1s.append(test_f1)

    print(f"Test Acc: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"Test F1 : {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")

if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True)
    parser.add_argument('-eeg_dir', type=str, required=True)
    parser.add_argument('-unfreeze_backbone', action='store_true')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-weight_decay', type=float, default=1e-2)
    parser.add_argument('-label_smoothing', type=float, default=0.1)
    parser.add_argument('-subjects_limit', type=int, default=0)
    args = parser.parse_args()
    main(args)
