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

# ================= 配置区 =================
MINI_BATCH_SIZE = 8
TARGET_BATCH_SIZE = 32
ACCUMULATION_STEPS = TARGET_BATCH_SIZE // MINI_BATCH_SIZE

# ================= 辅助类 =================
class VisualDataset(Dataset):
    def __init__(self, data_list, label_list, transform=None):
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        label_val = self.label_list[idx] # 这里 label_list 已经是展平后的 int 列表
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
        except:
            print(f"Warning: Failed to load {img_path}")
            img = torch.zeros(3, 224, 224)
        return img, torch.tensor(label_val, dtype=torch.long)

def get_visual_data_aligned(args):
    """
    读取数据并组织成 List[Subject] -> List[Trail] -> List[Samples] 的结构
    以便 LibEER 的 split.py 可以按照 Trail 进行划分 (Cross-Trail)
    """
    print(f"正在构建视觉数据集索引 (Cross-Trail Mode)...")
    all_labels = {}
    # 读取所有被试的标签文件
    for sub_id in range(1, 33):
        path = os.path.join(args.dataset_path, f"s{sub_id:02d}.dat")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                all_labels[sub_id] = content['labels']
        else:
            print(f"Warning: Label file for s{sub_id:02d} not found.")

    aligned_data = []  # 结构: [Subject_1_Trails, Subject_2_Trails, ...]
    aligned_label = [] # 结构: [Subject_1_Labels, Subject_2_Labels, ...]

    for sub_id in range(1, 33):
        sub_str = f"s{sub_id:02d}"
        sub_trails_data = []  # 存放该被试所有 Trail 的图片路径列表
        sub_trails_label = [] # 存放该被试所有 Trail 的标签列表
        
        sub_face_dir = os.path.join(args.faces_path, sub_str)
        
        if not os.path.exists(sub_face_dir) or sub_id not in all_labels:
            aligned_data.append([])
            aligned_label.append([])
            continue

        # DEAP 数据集通常有 40 个 Trail
        for trial_id in range(1, 41):
            # 获取该 Trail 的标签
            raw_label = all_labels[sub_id][trial_id - 1]
            valence, arousal = raw_label[0], raw_label[1]
            v_high = valence >= 5
            a_high = arousal >= 5
            
            # 4分类 (HALV) 逻辑
            if not v_high and not a_high: cls = 0   # LALV
            elif not v_high and a_high:   cls = 1   # LAHV
            elif v_high and not a_high:   cls = 2   # HALV
            elif v_high and a_high:       cls = 3   # HAHV
            
            # 获取该 Trail 下的所有 Segment 图片
            pattern = f"{sub_str}_trial{trial_id:02d}_seg*.jpg"
            search_path = os.path.join(sub_face_dir, pattern)
            files = glob.glob(search_path)
            # 按 segment ID 排序确保顺序
            files.sort(key=lambda x: int(x.split('_seg')[-1].split('.')[0]))
            
            if len(files) > 0:
                # 注意：这里我们将一个 Trail 的所有图片作为一个整体存入列表
                # split.py 会根据这个列表的长度（即 Trail 的数量）进行划分
                sub_trails_data.append(files)
                # 标签也要对应图片的数量，重复 cls
                sub_trails_label.append([cls] * len(files))

        aligned_data.append(sub_trails_data)
        aligned_label.append(sub_trails_label)
        print(f"Subject {sub_str}: Loaded {len(sub_trails_data)} trails.")
        
    return aligned_data, aligned_label

def flatten_data(data_trails, label_trails, indices):
    """
    辅助函数：将选中的 Trail 索引对应的图片和标签展平成一维列表
    """
    flat_data = []
    flat_label = []
    for i in indices:
        flat_data.extend(data_trails[i])
        flat_label.extend(label_trails[i])
    return flat_data, flat_label

# ================= 主程序 =================
def main(args):
    # 强制设置 split 相关的参数以符合你的要求
    if args.setting is None:
        # 如果没有指定 preset，手动构建一个基础 setting
        setting = set_setting_by_args(args)
    else:
        setting = preset_setting[args.setting](args)

    # 确保实验模式是 subject-dependent
    # 注意：split.py 依赖 setting 对象里的属性
    # 如果 args 里没有传这些参数，这里最好强制覆盖一下，或者确保命令行传入了正确的参数
    # 例如: --experiment_mode subject-dependent --split_type train-val-test
    
    setup_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Training on: {device} (Gradient Accumulation Mode)")

    # 1. 获取按 Trail 组织的数据
    data_all_subs, label_all_subs = get_visual_data_aligned(args)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_accuracies = [] # 记录所有被试在 Test 集上的最终准确率

    # 2. 遍历每个 Subject (Subject-Dependent)
    for rridx, (data_trails, label_trails) in enumerate(zip(data_all_subs, label_all_subs), 1):
        if len(data_trails) == 0:
            continue
        
        # 使用 LibEER 的划分逻辑
        # get_split_index 会返回 Trail 的索引 (因为传入的 data_trails 是 List[Trail])
        tts = get_split_index(data_trails, label_trails, setting)
        
        print(f"\n========== Subject {rridx} Training ==========")

        # 遍历划分 (通常 train-val-test 只有 1 个 fold，k-fold 会有多个)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            
            # 处理验证集为空的情况 (LibEER 逻辑)
            if val_indexes[0] == -1 or len(val_indexes) == 0:
                print("Notice: No validation set provided, using test set for validation.")
                val_indexes = test_indexes

            print(f"Fold {ridx} - Train Trails: {len(train_indexes)}, Val Trails: {len(val_indexes)}, Test Trails: {len(test_indexes)}")

            # 3. 将 Trail 索引展平为图片样本
            train_paths, train_lbls = flatten_data(data_trails, label_trails, train_indexes)
            val_paths, val_lbls = flatten_data(data_trails, label_trails, val_indexes)
            test_paths, test_lbls = flatten_data(data_trails, label_trails, test_indexes)

            # 构建 Dataset 和 DataLoader
            train_set = VisualDataset(train_paths, train_lbls, transform=train_transform)
            val_set = VisualDataset(val_paths, val_lbls, transform=val_test_transform)
            test_set = VisualDataset(test_paths, test_lbls, transform=val_test_transform)

            train_loader = DataLoader(train_set, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_set, batch_size=MINI_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=False)
            test_loader = DataLoader(test_set, batch_size=MINI_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=False)

            # 初始化模型
            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            model = fer.model
            
            # 修改分类头适配 4 分类
            num_ftrs = 1280
            try:
                if hasattr(model, 'num_features'): num_ftrs = model.num_features
                elif hasattr(model, 'classifier') and not isinstance(model.classifier, nn.Identity): num_ftrs = model.classifier.in_features
                elif hasattr(model, 'fc') and not isinstance(model.fc, nn.Identity): num_ftrs = model.fc.in_features
            except: pass
            
            model.classifier = nn.Linear(num_ftrs, 4)
            if hasattr(model, 'fc'): model.fc = nn.Identity()
            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            best_model_path = os.path.join(args.output_dir, f"visual_model_sub{rridx}_fold{ridx}_best.pth")

            # 🟢🟢🟢 训练循环 🟢🟢🟢
            for epoch in range(args.epochs):
                model.train()
                train_loss = 0
                
                # 训练步骤
                pbar = tqdm(train_loader, desc=f"Train S{rridx} F{ridx} Ep{epoch+1}", leave=False)
                for i, (imgs, targets) in enumerate(pbar):
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    
                    loss = loss / ACCUMULATION_STEPS
                    loss.backward()
                    
                    if (i + 1) % ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    train_loss += loss.item() * ACCUMULATION_STEPS

                # 验证步骤 (用于模型选择)
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs, targets = imgs.to(device), targets.to(device)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == targets.data)
                        val_total += targets.size(0)
                
                val_acc = val_correct.double() / val_total if val_total > 0 else 0
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_model_path)
                
                # print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f} Val Acc: {val_acc:.4f}")

            # 🟢🟢🟢 测试步骤 (Test Evaluation) 🟢🟢🟢
            # 加载验证集上表现最好的模型
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print(f"Loaded best model with Val Acc: {best_val_acc:.4f}")
            else:
                print("Warning: No model saved, using last epoch model.")

            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for imgs, targets in tqdm(test_loader, desc=f"Testing S{rridx} F{ridx}"):
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    test_correct += torch.sum(preds == targets.data)
                    test_total += targets.size(0)
            
            test_acc = test_correct.double() / test_total if test_total > 0 else 0
            print(f"👉 Subject {rridx} Fold {ridx} TEST ACCURACY: {test_acc:.4f}")
            test_accuracies.append(test_acc.item())

    print("\n========== Final Results ==========")
    print(f"Average Test Accuracy across {len(test_accuracies)} folds/subjects: {np.mean(test_accuracies):.4f}")

if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True, help='Path to the face images directory')
    # 建议运行时添加以下参数以确保 split 逻辑正确:
    # -experiment_mode subject-dependent -split_type train-val-test -test_size 0.2 -val_size 0.1
    args = parser.parse_args()
    
    args.output_dir = make_output_dir(args, "VisualModel")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"📂 Output directory: {args.output_dir}")
    
    main(args)