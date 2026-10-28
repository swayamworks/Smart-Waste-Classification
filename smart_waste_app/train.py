"""Train the Smart Waste Classifier (Phase 2 recipe) on TrashNet.

Usage (Colab / local):
    python train.py --data path/to/dataset-resized --out models/mobilenetv3_large_trashnet.pt

--data must contain one folder per class: cardboard, glass, metal, paper, plastic, trash
(TrashNet 'dataset-resized' from https://github.com/garythung/trashnet).

Recipe (from the reports): seed 42, 70/15/15 split at near-duplicate-group level,
AdamW (wd 0.01), class-weighted cross-entropy + label smoothing 0.10, batch 32,
2 frozen-backbone warm-up epochs (lr 1e-3) + up to 15 fine-tuning epochs
(backbone 1e-4, head 1e-3, cosine), early stopping on validation macro-F1 (patience 5).
"""
import argparse, hashlib, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.common import CLASSES, IMG_SIZE, MEAN, STD, build_model, eval_tf

SEED = 42


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data audit + split
def index_images(root: Path):
    items, bad = [], 0
    for ci, c in enumerate(CLASSES):
        for p in sorted((root / c).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                items.append((p, ci))
            except Exception:
                bad += 1
    print(f"indexed {len(items)} readable images ({bad} corrupted skipped)")
    return items


def drop_conflicting_duplicates(items):
    """Remove files that are byte-identical but carry different class labels."""
    by_hash = {}
    for p, y in items:
        by_hash.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append((p, y))
    keep, dropped = [], 0
    for grp in by_hash.values():
        if len({y for _, y in grp}) > 1:
            dropped += len(grp)
        else:
            keep.append(grp[0])           # also removes exact same-label duplicates
    print(f"removed {dropped} images duplicated under conflicting labels; {len(keep)} remain")
    return keep


def near_duplicate_groups(items, thr=0.98):
    """Group near-identical photos by 32x32 grayscale thumbnail correlation >= thr."""
    th = []
    for p, _ in items:
        with Image.open(p) as im:
            th.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32).ravel())
    X = np.stack(th); X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    C = X @ X.T / X.shape[1]
    parent = list(range(len(items)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in zip(*np.where(np.triu(C, 1) >= thr)):
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(items))]


def group_split(items, groups, fr=(0.70, 0.15, 0.15)):
    """Stratified (by class) 70/15/15 split where each group stays in one partition."""
    rng = random.Random(SEED)
    split = {}
    for ci in range(len(CLASSES)):
        gs = sorted({groups[i] for i, (_, y) in enumerate(items) if y == ci})
        rng.shuffle(gs)
        sizes = {g: sum(1 for i in range(len(items)) if groups[i] == g and items[i][1] == ci) for g in gs}
        total = sum(sizes.values()); acc = 0
        for g in gs:
            acc += sizes[g]
            split[(g, ci)] = 0 if acc <= fr[0] * total else (1 if acc <= (fr[0] + fr[1]) * total else 2)
    parts = ([], [], [])
    for i, it in enumerate(items):
        parts[split[(groups[i], it[1])]].append(it)
    return parts


class Imgs(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
    return np.array(ys), np.array(ps)


def run_epoch(model, loader, loss_fn, opt, scaler, dev):
    model.train(); tot = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y)
    return tot / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/mobilenetv3_large_trashnet.pt")
    ap.add_argument("--epochs", type=int, default=15, help="fine-tuning epochs (after 2 warm-up)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke test only)")
    a = ap.parse_args()
    seed_all(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    items = drop_conflicting_duplicates(index_images(Path(a.data)))
    groups = near_duplicate_groups(items)
    tr, va, te = group_split(items, groups)
    print(f"split -> train {len(tr)} / val {len(va)} / test {len(te)}")

    mk = lambda it, tf, sh: DataLoader(Imgs(it, tf), batch_size=a.batch, shuffle=sh, num_workers=2)
    L_tr, L_va, L_te = mk(tr, train_tf, True), mk(va, eval_tf, False), mk(te, eval_tf, False)

    counts = np.bincount([y for _, y in tr], minlength=len(CLASSES)).astype(float)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.10)

    model = build_model(pretrained=not a.no_pretrained).to(dev)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, bad = -1.0, None, 0

    def evaluate(tag, ep):
        nonlocal best, best_state, bad
        yv, pv = predict(model, L_va, dev); f1 = f1_score(yv, pv, average="macro")
        print(f"[{tag} {ep}] val acc {accuracy_score(yv, pv):.4f}  macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

    # stage 1: frozen backbone, train head only
    for p in model.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(1, a.warmup + 1):
        print(f"  loss {run_epoch(model, L_tr, loss_fn, opt, scaler, dev):.4f}"); evaluate("warm-up", ep)

    # stage 2: fine-tune everything, lower backbone LR, cosine schedule, early stopping
    for p in model.features.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": 1e-4},
                             {"params": model.classifier.parameters(), "lr": 1e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    bad = 0
    for ep in range(1, a.epochs + 1):
        print(f"  loss {run_epoch(model, L_tr, loss_fn, opt, scaler, dev):.4f}"); sched.step()
        evaluate("fine-tune", ep)
        if bad >= 5: print("early stopping"); break

    model.load_state_dict(best_state)              # restore best validation checkpoint
    yt, pt = predict(model, L_te, dev)
    rep = classification_report(yt, pt, target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {"test_accuracy": accuracy_score(yt, pt), "macro_precision": rep["macro avg"]["precision"],
               "macro_recall": rep["macro avg"]["recall"], "macro_f1": rep["macro avg"]["f1-score"],
               "best_val_macro_f1": best, "confusion_matrix": confusion_matrix(yt, pt).tolist(), "per_class": {c: rep[c] for c in CLASSES}}
    print(classification_report(yt, pt, target_names=CLASSES, digits=3, zero_division=0))
    print(json.dumps({k: v for k, v in metrics.items() if k in ("test_accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "classes": CLASSES, "arch": "mobilenet_v3_large", "val_macro_f1": best}, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
