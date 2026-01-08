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

from utils.args import get_args_parser
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, get_split_index, index_to_data
from utils.utils import setup_seed
from models.Models import Model as ModelDict


class PairedDataset(Dataset):
    """Pair EEG segment arrays with face image paths and per-segment labels.

    eeg_trials: list of trials, each trial -> list/ndarray of segments (C,L)
    face_trials: list of trials, each trial -> list of image paths (may be shorter)
    label_trials: list of trials, each trial -> per-segment labels (onehot or int)
    trial_indices: which trial indices to include
    """
    def __init__(self, eeg_trials, face_trials, label_trials, trial_indices, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        for t in trial_indices:
            if t < 0 or t >= len(eeg_trials):
                continue
            e_trial = eeg_trials[t]
            f_trial = face_trials[t] if t < len(face_trials) else []
            lab_trial = label_trials[t] if t < len(label_trials) else None
            if e_trial is None:
                continue
            num_segments = len(e_trial)
            for s in range(num_segments):
                eeg_seg = e_trial[s]
                img_path = f_trial[s] if s < len(f_trial) else None
                # determine label
                lbl = None
                if lab_trial is not None:
                    try:
                        l = lab_trial[s]
                        # if one-hot array
                        if hasattr(l, '__len__') and not isinstance(l, (int, np.integer)):
                            # convert to int index
                            try:
                                lbl = int(np.argmax(l))
                            except Exception:
                                lbl = int(l[0]) if len(l)>0 else 0
                        else:
                            lbl = int(l)
                    except Exception:
                        lbl = 0
                else:
                    lbl = 0
                self.samples.append((eeg_seg.astype('float32'), img_path, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg_seg, img_path, lbl = self.samples[idx]
        eeg_tensor = torch.tensor(eeg_seg)
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
            except Exception:
                img = torch.zeros(3,224,224)
        else:
            img = torch.zeros(3,224,224)
        return eeg_tensor, img, int(lbl)


def build_face_index(dataset_path, faces_path):
    """Return per-subject list: face_trials[sub_idx-1] -> list of trials -> list of image paths"""
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
    # support 'hsemotion' (HSEmotionRecognizer) or torchvision resnet
    if name == 'hsemotion':
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')
            model = fer.model
            # adapt classifier
            num_ftrs = 1280
            try:
                if hasattr(model, 'num_features'): num_ftrs = model.num_features
                elif hasattr(model, 'classifier') and not isinstance(model.classifier, nn.Identity): num_ftrs = model.classifier.in_features
                elif hasattr(model, 'fc') and not isinstance(model.fc, nn.Identity): num_ftrs = model.fc.in_features
            except: pass
            model.classifier = nn.Linear(num_ftrs, num_classes)
            if hasattr(model, 'fc'): model.fc = nn.Identity()
            return model.to(device)
        except Exception:
            print('HSEmotion not available, falling back to resnet18')
    # fallback resnet18
    import torchvision.models as models
    res = models.resnet18(pretrained=False)
    res.fc = nn.Linear(res.fc.in_features, num_classes)
    return res.to(device)


def instantiate_eeg_model(name, channels, feature_dim, num_classes, device):
    # use Models dict if available
    if name in ModelDict:
        return ModelDict[name](channels, feature_dim, num_classes).to(device)
    # fallback simple model
    class SimpleEEG(nn.Module):
        def __init__(self, c, L, nc):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(c,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(32*16, nc))
        def forward(self,x):
            return self.net(x)
    return SimpleEEG(channels, feature_dim, num_classes).to(device)


def get_encoder_from_model(model):
    """Build a callable encoder from a model.

    Strategy:
    - Recursively expand any `ModuleList` or `Sequential` children so we don't include container objects
      (ModuleList has no `forward`).
    - Remove a final classifier `nn.Linear` layer if present (common pattern: ... -> avgpool -> fc).
    - Return an `nn.Sequential` of the remaining feature extraction layers followed by `nn.Flatten()`.
    This aims to be robust across different model layouts used in the repository.
    """
    def _expand(mod, out_list):
        # If container, expand its contents; otherwise append the module
        if isinstance(mod, (nn.Sequential, nn.ModuleList)):
            for m in mod:
                _expand(m, out_list)
        else:
            out_list.append(mod)

    children = list(model.children())
    flat = []
    for c in children:
        _expand(c, flat)

    # remove trailing classifier if it's a Linear layer
    if len(flat) > 0 and isinstance(flat[-1], nn.Linear):
        flat = flat[:-1]

    # If nothing left, fallback to identity flatten
    if len(flat) == 0:
        return nn.Sequential(nn.Flatten())

    # Ensure all elements are modules with forward (we expanded containers above)
    try:
        enc = nn.Sequential(*flat, nn.Flatten())
        # quick sanity check: ensure it's callable
        _ = list(enc.children())
        return enc
    except Exception:
        return nn.Sequential(nn.Flatten())


def adjust_lr_by_val(optimizer, acc_eeg, acc_vis, base_lr, min_lr=1e-6, max_lr=1e-2):
    # simple heuristic: increase lr of weaker modality by 10% up to max, decrease stronger by 10%
    if acc_eeg is None or acc_vis is None:
        return
    for group in optimizer.param_groups:
        tag = group.get('tag', None)
        if tag == 'eeg':
            if acc_eeg > acc_vis + 0.02:
                group['lr'] = max(min_lr, group['lr'] * 0.9)
            elif acc_vis > acc_eeg + 0.02:
                group['lr'] = min(max_lr, group['lr'] * 1.1)
        elif tag == 'vis':
            if acc_vis > acc_eeg + 0.02:
                group['lr'] = max(min_lr, group['lr'] * 0.9)
            elif acc_eeg > acc_vis + 0.02:
                group['lr'] = min(max_lr, group['lr'] * 1.1)


def main():
    parser = get_args_parser()
    parser.add_argument('-faces_path', type=str, required=True)
    parser.add_argument('-eeg_model', type=str, default='EEGNet')
    parser.add_argument('-vis_model', type=str, default='hsemotion')
    parser.add_argument('-fusion', type=str, default='late', choices=['late','concat'])
    args = parser.parse_args()
    # default bounds for DEAP if not provided
    if getattr(args, 'bounds', None) is None:
        args.bounds = [5, 5]

    if args.setting is None:
        setting = set_setting_by_args(args)
    else:
        setting = preset_setting[args.setting](args)

    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load data and merge
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)

    # build face index mapping
    face_index = build_face_index(args.dataset_path, args.faces_path)

    # instantiate models
    eeg_model = instantiate_eeg_model(args.eeg_model, channels, feature_dim, num_classes, device)
    vis_model = instantiate_visual_model(args.vis_model, device, num_classes)

    # build encoders (feature-level)
    eeg_encoder = get_encoder_from_model(eeg_model).to(device)
    vis_encoder = get_encoder_from_model(vis_model).to(device)

    # fusion param (learnable mixing for late fusion)
    mix_param = nn.Parameter(torch.tensor(0.0)).to(device)

    # infer feature dims for fusion head
    with torch.no_grad():
        # EEG models often expect input shape (B, 1, channels, sample_length)
        d_eeg = torch.zeros(1, 1, channels, args.sample_length).to(device)
        try:
            fe = eeg_encoder(d_eeg)
            feat_eeg_dim = int(fe.shape[1])
        except Exception:
            feat_eeg_dim = 256
        d_img = torch.zeros(1, 3, 224, 224).to(device)
        try:
            fv = vis_encoder(d_img)
            feat_vis_dim = int(fv.shape[1])
        except Exception:
            feat_vis_dim = 256

    fusion_head = nn.Sequential(nn.Linear(feat_eeg_dim + feat_vis_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)).to(device)

    # optimizer with modality-tagged groups
    params = [
        {'params': eeg_encoder.parameters(), 'lr': args.lr, 'tag': 'eeg'},
        {'params': vis_encoder.parameters(), 'lr': args.lr, 'tag': 'vis'},
        {'params': fusion_head.parameters(), 'lr': args.lr, 'tag': 'fusion'},
        {'params': [mix_param], 'lr': args.lr}
    ]
    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()

    # iterate subjects
    for sub_idx, (data_sub, label_sub) in enumerate(zip(data, label), 1):
        if len(data_sub) == 0:
            continue
        print(f"Subject {sub_idx}: preparing folds...")
        tts = get_split_index(data_sub, label_sub, setting)
        for ridx, (train_idx, test_idx, val_idx) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            print(f"Subject {sub_idx} Fold {ridx}: train {len(train_idx)} trials, val {len(val_idx)} trials, test {len(test_idx)} trials")
            # print exact trial indices for audit
            try:
                print(f" Subject {sub_idx} Fold {ridx} trial indices -> TRAIN: {train_idx}, VAL: {val_idx if val_idx[0] != -1 else []}, TEST: {test_idx}")
            except Exception:
                pass

            # get keep-dim data for pairing
            tr_data, tr_label, val_data, val_label, te_data, te_label = index_to_data(data_sub, label_sub, train_idx, test_idx, val_idx, keep_dim=True)

            # face trials for this subject (face_index is 0-based subjects)
            face_trials_all = face_index[sub_idx-1] if sub_idx-1 < len(face_index) else []

            # if subject has no face images at all, skip multimodal training for this subject
            has_any_face = any((isinstance(t, (list, tuple)) and len(t) > 0) for t in face_trials_all)
            if not has_any_face:
                print(f"Subject {sub_idx} has no face images under faces_path; skipping subject for multimodal training.")
                continue

            # build aligned face lists for train/val/test matching returned tr/val/te lists
            face_train = [face_trials_all[i] if i < len(face_trials_all) else [] for i in train_idx]
            face_val = [face_trials_all[i] if i < len(face_trials_all) else [] for i in val_idx] if val_idx[0] != -1 else []
            face_test = [face_trials_all[i] if i < len(face_trials_all) else [] for i in test_idx]

            # build paired datasets: tr_data is list-of-trials aligned with face_train list
            ds_train = PairedDataset(tr_data, face_train, tr_label, list(range(len(tr_data))))
            ds_val = PairedDataset(val_data, face_val, val_label, list(range(len(val_data))))
            ds_test = PairedDataset(te_data, face_test, te_label, list(range(len(te_data))))

            loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
            loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
            loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

            best_val = 0.0
            best_state = None
            for epoch in range(args.epochs):
                eeg_encoder.train(); vis_encoder.train(); fusion_head.train();
                total = 0
                correct = 0
                for batch in loader_train:
                    if len(batch) == 3:
                        eegs, imgs, targets = batch
                    else:
                        continue
                    # preserve original eegs with batch dim: (B, C, L)
                    if eegs.dim() == 2:
                        orig_eegs = eegs.unsqueeze(0).to(device)
                    else:
                        orig_eegs = eegs.to(device)
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    # prepare encoder input if possible: (B,1,C,L)
                    try:
                        eegs_for_encoder = orig_eegs.unsqueeze(1)
                    except Exception:
                        eegs_for_encoder = None

                    optimizer.zero_grad()
                    if args.fusion == 'concat':
                        # prefer feature-level encoder; fallback to logits if encoder incompatible
                        if eegs_for_encoder is not None:
                            try:
                                feat_e = eeg_encoder(eegs_for_encoder)
                            except Exception:
                                feat_e = eeg_model(orig_eegs)
                        else:
                            feat_e = eeg_model(orig_eegs)
                        feat_v = vis_encoder(imgs)
                        feat = torch.cat([feat_e, feat_v], dim=1)
                        logits = fusion_head(feat)
                    else:
                        outputs_eeg = eeg_model(orig_eegs)
                        outputs_vis = vis_model(imgs)
                        alpha = torch.sigmoid(mix_param)
                        logits = alpha * outputs_eeg + (1 - alpha) * outputs_vis

                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()

                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

                train_acc = correct / total if total > 0 else 0.0

                # validation
                eeg_encoder.eval(); vis_encoder.eval(); fusion_head.eval();
                val_total = 0
                val_correct = 0
                val_correct_eeg = 0
                val_correct_vis = 0
                with torch.no_grad():
                    for batch in loader_val:
                        if len(batch) != 3:
                            continue
                        eegs, imgs, targets = batch
                        # preserve original eegs with batch dim
                        if eegs.dim() == 2:
                            orig_eegs = eegs.unsqueeze(0).to(device)
                        else:
                            orig_eegs = eegs.to(device)
                        imgs = imgs.to(device)
                        targets = targets.to(device)
                        try:
                            eegs_for_encoder = orig_eegs.unsqueeze(1)
                        except Exception:
                            eegs_for_encoder = None
                        if args.fusion == 'concat':
                            if eegs_for_encoder is not None:
                                try:
                                    feat_e = eeg_encoder(eegs_for_encoder)
                                except Exception:
                                    feat_e = eeg_model(orig_eegs)
                            else:
                                feat_e = eeg_model(orig_eegs)
                            feat_v = vis_encoder(imgs)
                            feat = torch.cat([feat_e, feat_v], dim=1)
                            logits = fusion_head(feat)
                            outputs_eeg = eeg_model(orig_eegs)
                            outputs_vis = vis_model(imgs)
                        else:
                            outputs_eeg = eeg_model(orig_eegs)
                            outputs_vis = vis_model(imgs)
                            alpha = torch.sigmoid(mix_param)
                            logits = alpha * outputs_eeg + (1 - alpha) * outputs_vis

                        preds = logits.argmax(dim=1)
                        val_correct += (preds == targets).sum().item()
                        val_total += targets.size(0)

                        # modality accs
                        val_correct_eeg += (outputs_eeg.argmax(dim=1) == targets).sum().item()
                        val_correct_vis += (outputs_vis.argmax(dim=1) == targets).sum().item()

                val_acc = val_correct / val_total if val_total > 0 else 0.0
                val_acc_eeg = val_correct_eeg / val_total if val_total > 0 else None
                val_acc_vis = val_correct_vis / val_total if val_total > 0 else None

                # adjust lrs heuristically
                adjust_lr_by_val(optimizer, val_acc_eeg, val_acc_vis, args.lr)

                print(f"Subject {sub_idx} Fold {ridx} Epoch {epoch+1} TrainAcc {train_acc:.4f} ValAcc {val_acc:.4f}")

                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {
                        'eeg_encoder': eeg_encoder.state_dict(),
                        'vis_encoder': vis_encoder.state_dict(),
                        'fusion_head': fusion_head.state_dict(),
                        'eeg_model': eeg_model.state_dict(),
                        'vis_model': vis_model.state_dict(),
                        'mix_param': mix_param.detach().cpu().numpy()
                    }

            # test using best_state if available
            if best_state is not None:
                eeg_encoder.load_state_dict(best_state['eeg_encoder'])
                vis_encoder.load_state_dict(best_state['vis_encoder'])
                fusion_head.load_state_dict(best_state['fusion_head'])
                eeg_model.load_state_dict(best_state['eeg_model'])
                vis_model.load_state_dict(best_state['vis_model'])

            # final test
            eeg_encoder.eval(); vis_encoder.eval(); fusion_head.eval();
            test_total = 0
            test_correct = 0
            with torch.no_grad():
                for batch in loader_test:
                    if len(batch) != 3:
                        continue
                    eegs, imgs, targets = batch
                    if eegs.dim() == 2:
                        orig_eegs = eegs.unsqueeze(0).to(device)
                    else:
                        orig_eegs = eegs.to(device)
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    try:
                        eegs_for_encoder = orig_eegs.unsqueeze(1)
                    except Exception:
                        eegs_for_encoder = None
                    if args.fusion == 'concat':
                        if eegs_for_encoder is not None:
                            try:
                                feat_e = eeg_encoder(eegs_for_encoder)
                            except Exception:
                                feat_e = eeg_model(orig_eegs)
                        else:
                            feat_e = eeg_model(orig_eegs)
                        feat_v = vis_encoder(imgs)
                        feat = torch.cat([feat_e, feat_v], dim=1)
                        logits = fusion_head(feat)
                    else:
                        outputs_eeg = eeg_model(orig_eegs)
                        outputs_vis = vis_model(imgs)
                        alpha = torch.sigmoid(mix_param)
                        logits = alpha * outputs_eeg + (1 - alpha) * outputs_vis
                    preds = logits.argmax(dim=1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.size(0)

            test_acc = test_correct / test_total if test_total > 0 else 0.0
            print(f"Subject {sub_idx} Fold {ridx} TEST Acc {test_acc:.4f}")


if __name__ == '__main__':
    main()
