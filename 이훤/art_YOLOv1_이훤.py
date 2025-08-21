# -*- coding: utf-8 -*-
# yolov1_trainval_json.py
# YOLOv1-style (S=7, B=2) detector for 1280x1280 grayscale drawings -> 448x448
# Dataset layout:
# data/
#   Training/
#     labeling_data/{house,man,tree,woman}/*.json
#     origin_data/{house,man,tree,woman}/*.(jpg|png|bmp...)
#   Validation/
#     labeling_data/{house,man,tree,woman}/*.json
#     origin_data/{house,man,tree,woman}/*.(jpg|png|bmp...)

from pathlib import Path
import os, random, json
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def imread_unicode(path, flags):
    import numpy as np, cv2
    path = str(path)  # Path -> str
    try:
        # 윈도우/유니코드 경로 안전하게 바이너리로 읽어서 디코딩
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


# -------------------------
# Config
# -------------------------
DATA_ROOT = Path("./data")
TRAIN_SPLIT = DATA_ROOT / "Training"
VAL_SPLIT   = DATA_ROOT / "Validation"

ORIGIN_DIR_NAME = "origin_data"     # 주신 구조 사용
LABEL_DIR_NAME  = "labeling_data"

CLASSES = ["house", "woman", "man", "tree"]
C = len(CLASSES)
S = 7
B = 2
IMG_SIZE = 448
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

# -------------------------
# Korean label mapping
# -------------------------
EXCLUDE_PARTS = {"지붕","집벽","문","창문","굴뚝","연기"}  # 집의 부분 요소는 제외
LABEL_RULES = [
    ("house", ["집전체","집"]),
    ("woman", ["여자사람","여자"]),
    ("man",   ["남자사람","남자"]),
    ("tree",  ["나무"]),
]
def map_korean_label_to_class_id(lbl: str):
    if not isinstance(lbl, str):
        return None
    for excl in EXCLUDE_PARTS:
        if excl in lbl:
            return None
    for cname, keys in LABEL_RULES:
        for k in keys:
            if k in lbl:
                return CLASSES.index(cname)
    return None

# -------------------------
# Utils
# -------------------------
def iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps=1e-9):
    x1 = torch.max(a[...,0], b[...,0])
    y1 = torch.max(a[...,1], b[...,1])
    x2 = torch.min(a[...,2], b[...,2])
    y2 = torch.min(a[...,3], b[...,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (a[...,2]-a[...,0]).clamp(min=0) * (a[...,3]-a[...,1]).clamp(min=0)
    area2 = (b[...,2]-b[...,0]).clamp(min=0) * (b[...,3]-b[...,1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union

def nms(boxes, scores, iou_thr=0.5):
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        iou = iou_xyxy(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][iou <= iou_thr]
    return keep

# -------------------------
# Dataset for one split (Training or Validation)
# -------------------------
class YoloJSONSplitDataset(Dataset):
    """
    split_root/
      origin_data/{house,man,tree,woman}/*.(jpg|...)
      labeling_data/{house,man,tree,woman}/*.json

    JSON schema example (from user):
    {
      "meta": {..., "img_resolution": "1280x1280", ...},
      "annotations": {
        "bbox": [ {"label": "집전체", "x": 356, "y": 178, "w": 545, "h": 742}, ... ]
      }
    }

    - 픽셀 좌표(x,y,w,h: top-left) -> center 정규화(cx,cy,w,h).
    - 4개 클래스만 사용, 나머지는 무시.
    - 셀당 1개 GT만 유지(면적 큰 박스 우선).
    """
    def __init__(self, split_root: Path, classes: List[str], s=7,
                 img_size=IMG_SIZE, keep_unlabeled=False, debug_unknown=False):
        self.split_root = split_root
        self.classes = classes
        self.s = s
        self.img_size = img_size
        self.keep_unlabeled = keep_unlabeled
        self.debug_unknown = debug_unknown

        self.img_dir = split_root / ORIGIN_DIR_NAME
        self.lbl_dir = split_root / LABEL_DIR_NAME

        if not self.img_dir.exists():
            raise RuntimeError(f"Image dir not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise RuntimeError(f"Label dir not found: {self.lbl_dir}")

        self.items = []
        for cname in classes:
            cdir = self.img_dir / cname
            if not cdir.exists(): 
                continue
            for p in cdir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.items.append(p)
        self.items.sort()
        if not self.items:
            raise RuntimeError(f"No images under {self.img_dir}")

    def __len__(self): return len(self.items)

    def _parse_label_json(self, jpath: Path, img_w: int, img_h: int):
        out = []
        if not jpath.exists():
            return out
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] bad json: {jpath} ({e})")
            return out

        ann = obj.get("annotations", {})
        bboxes = ann.get("bbox", [])
        if not isinstance(bboxes, list):
            return out

        for b in bboxes:
            if not isinstance(b, dict): 
                continue
            lbl = b.get("label", "")
            cid = map_korean_label_to_class_id(lbl)
            if cid is None:
                if self.debug_unknown:
                    print(f"[SKIP] label '{lbl}' not used @ {jpath.name}")
                continue
            try:
                x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            except Exception:
                continue
            if w <= 0 or h <= 0: 
                continue

            # pixel -> normalized
            x /= img_w; y /= img_h; w /= img_w; h /= img_h
            cx = x + w/2.0; cy = y + h/2.0
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            w  = float(np.clip(w,  1e-6, 1.0))
            h  = float(np.clip(h,  1e-6, 1.0))
            out.append((cid, cx, cy, w, h))
        return out

    def _load_all_labels_for_stem(self, stem: str, img_w: int, img_h: int):
        # 4개 클래스 폴더의 동일 stem.json을 모두 모아 한 이미지의 GT로 사용
        gts = []
        for cname in self.classes:
            jpath = self.lbl_dir / cname / f"{stem}.json"
            gts += self._parse_label_json(jpath, img_w, img_h)
        return gts

    def _encode_target(self, gts: List[Tuple[int,float,float,float,float]]):
        tgt = np.zeros((self.s, self.s, 5 + len(self.classes)), dtype=np.float32)
        chosen = {}
        for cls_id, cx,cy,w,h in gts:
            i = min(int(cx * self.s), self.s-1)
            j = min(int(cy * self.s), self.s-1)
            area = w*h
            key = (j,i)
            if key in chosen and area <= chosen[key][0]:
                continue
            chosen[key] = (area, cls_id, cx,cy,w,h)

        for (j,i), (_, cls_id, cx,cy,w,h) in chosen.items():
            cell_cx = cx * self.s - i
            cell_cy = cy * self.s - j
            tgt[j,i,0] = 1.0
            tgt[j,i,1:5] = [np.clip(cell_cx,0,1), np.clip(cell_cy,0,1), w, h]
            tgt[j,i,5 + cls_id] = 1.0
        return tgt

    def __getitem__(self, idx):
        ipath = self.items[idx]
        img0 = imread_unicode(ipath, cv2.IMREAD_GRAYSCALE)
        # __getitem__에서 읽기 실패 시
        if img0 is None:
            # 읽기 실패 파일 스킵하고 다음 샘플로
            return self.__getitem__((idx + 1) % len(self.items))
        if img0 is None:
            raise FileNotFoundError(f"Bad image: {ipath}")
        h0, w0 = img0.shape[:2]

        stem = ipath.stem
        gts = self._load_all_labels_for_stem(stem, w0, h0)
        if not gts and not self.keep_unlabeled:
            return self.__getitem__((idx+1) % len(self.items))

        # 간단 증강: 좌우 플립
        if random.random() < 0.5:
            img0 = cv2.flip(img0, 1)
            gts = [(cid, 1.0-cx, cy, w, h) for (cid,cx,cy,w,h) in gts]

        img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        img = img[None, ...]  # (1,H,W)
        target = self._encode_target(gts)
        return torch.from_numpy(img), torch.from_numpy(target), str(ipath)

# -------------------------
# Model (YOLOv1-like)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class YoloV1Backbone(nn.Module):
    """448 -> 7 via 6x MaxPool(2,2)"""
    def __init__(self, in_ch=1):
        super().__init__()
        chs = [64,128,256,512,1024,1024]
        layers, c_in = [], in_ch
        for c_out in chs:
            layers += [ConvBlock(c_in, c_out, 3, 1, 1),
                       ConvBlock(c_out, c_out, 3, 1, 1),
                       nn.MaxPool2d(2,2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)  # (N,1024,7,7)
    def forward(self, x): return self.features(x)

class YoloV1Head(nn.Module):
    def __init__(self, s=7, b=2, c=4, in_ch=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_ch * s * s, 4096)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, s*s*(b*5 + c))
        self.s, self.b, self.c = s,b,c
    def forward(self, feat):
        n = feat.size(0)
        x = feat.view(n, -1)
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x.view(n, self.s, self.s, self.b*5 + self.c)

class YoloV1(nn.Module):
    def __init__(self, s=7, b=2, c=4, in_ch=1):
        super().__init__()
        self.backbone = YoloV1Backbone(in_ch=in_ch)
        self.head = YoloV1Head(s,b,c,in_ch=1024)
    def forward(self, x): return self.head(self.backbone(x))

# -------------------------
# Loss (YOLOv1)
# -------------------------
class YoloV1Loss(nn.Module):
    def __init__(self, s=7, b=2, c=4, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.s, self.b, self.c = s,b,c
        self.lc = lambda_coord
        self.lno = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        """
        pred: (N,S,S, B*5 + C) -> [b1(5), b2(5), class(C)]
        target: (N,S,S, 5 + C) -> [obj, x,y,w,h, onehot(C)]
        """
        N,S,B,C = pred.size(0), self.s, self.b, self.c
        box1 = pred[...,0:5]
        box2 = pred[...,5:10]
        class_logits = pred[...,10:]

        sig = torch.sigmoid
        b1_xywh = sig(box1[...,0:4]); b1_conf = sig(box1[...,4:5])
        b2_xywh = sig(box2[...,0:4]); b2_conf = sig(box2[...,4:5])
        class_prob = torch.softmax(class_logits, dim=-1)

        obj = target[...,0:1]
        t_xywh = target[...,1:5]
        t_class = target[...,5:]

        gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        gx = gx.to(pred.device).float().view(1,S,S,1)
        gy = gy.to(pred.device).float().view(1,S,S,1)

        def to_abs_xyxy(xywh):
            cx = (gx + xywh[...,0:1]) / S
            cy = (gy + xywh[...,1:2]) / S
            w  = xywh[...,2:3]; h = xywh[...,3:4]
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            return torch.cat([x1,y1,x2,y2], dim=-1)

        b1_xyxy = to_abs_xyxy(b1_xywh)
        b2_xyxy = to_abs_xyxy(b2_xywh)
        t_cx = (gx + t_xywh[...,0:1]) / S
        t_cy = (gy + t_xywh[...,1:2]) / S
        t_w  = t_xywh[...,2:3]; t_h = t_xywh[...,3:4]
        t_xyxy = torch.cat([t_cx - t_w/2, t_cy - t_h/2, t_cx + t_w/2, t_cy + t_h/2], dim=-1)

        iou1 = iou_xyxy(b1_xyxy, t_xyxy)
        iou2 = iou_xyxy(b2_xyxy, t_xyxy)
        resp1 = (iou1 >= iou2).float().unsqueeze(-1) * obj
        resp2 = (iou2 >  iou1).float().unsqueeze(-1) * obj

        # coord (√w, √h)
        pred_xy_1 = b1_xywh[...,0:2]
        pred_wh_1 = torch.sqrt(b1_xywh[...,2:4].clamp(min=1e-6))
        targ_xy   = t_xywh[...,0:2]
        targ_wh   = torch.sqrt(t_xywh[...,2:4].clamp(min=1e-6))
        coord1 = nn.functional.mse_loss(resp1 * (pred_xy_1 - targ_xy), torch.zeros_like(targ_xy), reduction="sum") \
               + nn.functional.mse_loss(resp1 * (pred_wh_1 - targ_wh), torch.zeros_like(targ_wh), reduction="sum")
        pred_xy_2 = b2_xywh[...,0:2]
        pred_wh_2 = torch.sqrt(b2_xywh[...,2:4].clamp(min=1e-6))
        coord2 = nn.functional.mse_loss(resp2 * (pred_xy_2 - targ_xy), torch.zeros_like(targ_xy), reduction="sum") \
               + nn.functional.mse_loss(resp2 * (pred_wh_2 - targ_wh), torch.zeros_like(targ_wh), reduction="sum")
        coord_loss = self.lc * (coord1 + coord2)

        # confidence
        obj_conf1 = iou1.unsqueeze(-1) * obj
        obj_conf2 = iou2.unsqueeze(-1) * obj
        conf_obj_loss = nn.functional.mse_loss(resp1 * (b1_conf - obj_conf1), torch.zeros_like(obj_conf1), reduction="sum") \
                      + nn.functional.mse_loss(resp2 * (b2_conf - obj_conf2), torch.zeros_like(obj_conf2), reduction="sum")

        noobj = 1.0 - obj
        conf_noobj_loss = self.lno * (
            nn.functional.mse_loss(noobj * b1_conf, torch.zeros_like(b1_conf), reduction="sum") +
            nn.functional.mse_loss(noobj * b2_conf, torch.zeros_like(b2_conf), reduction="sum")
        )

        class_loss = nn.functional.mse_loss(obj * (class_prob - t_class), torch.zeros_like(t_class), reduction="sum")

        loss = (coord_loss + conf_obj_loss + conf_noobj_loss + class_loss) / max(1, N)
        return loss, {
            "coord": coord_loss.detach()/max(1,N),
            "conf_obj": conf_obj_loss.detach()/max(1,N),
            "conf_noobj": conf_noobj_loss.detach()/max(1,N),
            "class": class_loss.detach()/max(1,N),
        }

# -------------------------
# Train / Val / Infer helpers
# -------------------------
def make_loader(split_root: Path, batch=8, workers=0, keep_unlabeled=False, debug_unknown=False, shuffle=False, drop_last=False):
    ds = YoloJSONSplitDataset(split_root, CLASSES, s=S, img_size=IMG_SIZE,
                              keep_unlabeled=keep_unlabeled, debug_unknown=debug_unknown)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=drop_last)

def train_one_epoch(model, crit, loader, opt, epoch, log_every=50):
    model.train()
    running = 0.0
    for it, (imgs, tgts, _) in enumerate(loader):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        pred = model(imgs)
        loss, parts = crit(pred, tgts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()
        if (it+1) % log_every == 0:
            print(f"[E{epoch} I{it+1}] loss={loss.item():.4f} "
                  f"coord={parts['coord']:.3f} obj={parts['conf_obj']:.3f} "
                  f"noobj={parts['conf_noobj']:.3f} cls={parts['class']:.3f}")
    return running / max(1,len(loader))

@torch.no_grad()
def validate(model, crit, loader):
    model.eval()
    tot, n = 0.0, 0
    for imgs, tgts, _ in loader:
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        pred = model(imgs)
        loss, _ = crit(pred, tgts)
        tot += loss.item(); n += 1
    return tot / max(1,n)

@torch.no_grad()
def infer_on_image(model, img_path: Path, conf_thr=0.25, nms_iou=0.5):
    model.eval()
    img0 = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
    H,W = img0.shape[:2]
    img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.0
    out = model(torch.from_numpy(img[None,None,...]).to(DEVICE)).squeeze(0).cpu()

    box1 = out[...,0:5]; box2 = out[...,5:10]; cls  = out[...,10:]
    sig = torch.sigmoid
    b1_xywh = sig(box1[...,0:4]); b1_conf = sig(box1[...,4])
    b2_xywh = sig(box2[...,0:4]); b2_conf = sig(box2[...,4])
    cls_prob = torch.softmax(cls, dim=-1)

    gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")

    def decode_abs(xywh):
        cx = (gx.float() + xywh[...,0]) / S
        cy = (gy.float() + xywh[...,1]) / S
        w  = xywh[...,2]; h = xywh[...,3]
        x1 = (cx - w/2).clamp(0,1); y1 = (cy - h/2).clamp(0,1)
        x2 = (cx + w/2).clamp(0,1); y2 = (cy + h/2).clamp(0,1)
        return torch.stack([x1,y1,x2,y2], dim=-1)

    boxes_all, scores_all, labels_all = [], [], []
    for b_xywh, b_conf in [(b1_xywh, b1_conf), (b2_xywh, b2_conf)]:
        boxes = decode_abs(b_xywh).view(S*S,4)
        scores = (b_conf.view(S*S,1) * cls_prob.view(S*S, C))
        for c in range(C):
            sc = scores[:,c]
            m = sc >= conf_thr
            if m.sum()==0: continue
            boxes_c, scores_c = boxes[m], sc[m]
            keep = nms(boxes_c, scores_c, iou_thr=nms_iou)
            for k in keep:
                boxes_all.append(boxes_c[k]); scores_all.append(scores_c[k]); labels_all.append(c)

    if len(boxes_all)==0: return []
    boxes_all = torch.stack(boxes_all); scores_all = torch.stack(scores_all); labels_all = torch.tensor(labels_all)

    x1 = (boxes_all[:,0] * W).clamp(0,W-1).int().tolist()
    y1 = (boxes_all[:,1] * H).clamp(0,H-1).int().tolist()
    x2 = (boxes_all[:,2] * W).clamp(0,W-1).int().tolist()
    y2 = (boxes_all[:,3] * H).clamp(0,H-1).int().tolist()
    dets = []
    for i in range(len(x1)):
        dets.append({"xyxy": (x1[i],y1[i],x2[i],y2[i]),
                     "score": float(scores_all[i]),
                     "class_id": int(labels_all[i]),
                     "class_name": CLASSES[int(labels_all[i])]})
    return dets

def draw_and_save(img_path: Path, dets, save_path: Path):
    im = imread_unicode(img_path, cv2.IMREAD_COLOR)
    if im is None: 
        return
    for d in dets:
        x1,y1,x2,y2 = d["xyxy"]; cls = d["class_name"]; sc = d["score"]
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(im, f"{cls}:{sc:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), im)

def sample_val_vis(model):
    # Validation split에서 한 장 골라 시각화
    cand = []
    for cname in CLASSES:
        cand += [p for p in (VAL_SPLIT/ORIGIN_DIR_NAME/cname).rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not cand:
        print("[viz] No images in Validation.")
        return
    img_path = random.choice(cand)
    dets = infer_on_image(model, img_path, conf_thr=0.25, nms_iou=0.5)
    out_path = Path("runs/val_pred.jpg")
    draw_and_save(img_path, dets, out_path)
    print(f"[viz] {img_path.name} -> {out_path} | dets={len(dets)}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 1) Dataloaders
    dl_tr = make_loader(TRAIN_SPLIT, batch=8, workers=0, keep_unlabeled=False, debug_unknown=False, shuffle=True,  drop_last=True)
    dl_va = make_loader(VAL_SPLIT,   batch=8, workers=0, keep_unlabeled=False, debug_unknown=False, shuffle=False, drop_last=False)

    # 2) Model / Loss / Optimizer
    model = YoloV1(s=S, b=B, c=C, in_ch=1).to(DEVICE)
    criterion = YoloV1Loss(s=S, b=B, c=C, lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3) Train
    EPOCHS = 1
    best = 1e9
    Path("runs").mkdir(exist_ok=True)
    for e in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, criterion, dl_tr, opt, epoch=e, log_every=20)
        va_loss = validate(model, criterion, dl_va)
        print(f"[Epoch {e}] train={tr_loss:.4f} | val={va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "epoch": e}, Path("runs/yolov1_4cls.pt"))
            print("  -> saved runs/yolov1_4cls.pt")

    # 4) Quick visual check on Validation
    sample_val_vis(model)