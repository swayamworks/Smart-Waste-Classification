"""Smart Waste Classification API (FastAPI).

POST /predict   -> class, confidence, sorting stream, top-3, auto-sort vs manual-review
GET  /health    -> model status
GET  /          -> web frontend
Run:  uvicorn app.main:app --reload
"""
import base64, io, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .common import CLASSES, CONF_THRESHOLD, STREAMS, build_model, eval_tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "mobilenetv3_large_trashnet.pt"))
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Smart Waste Classification", version="1.0")
state = {"model": None, "error": None, "classes": CLASSES}


def load_model():
    """Load the trained checkpoint (state_dict, {'model_state'|'state_dict'|'model': ...}, or full model)."""
    if not MODEL_PATH.exists():
        state["error"] = f"Model file not found at {MODEL_PATH}. Train with train.py or copy your .pt checkpoint there."
        return
    try:
        obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            model = obj
        else:
            sd = obj
            if isinstance(obj, dict):
                for k in ("model_state", "state_dict", "model"):
                    if k in obj and isinstance(obj[k], dict):
                        sd = obj[k]; break
                if "classes" in obj and list(obj["classes"]) != CLASSES:
                    state["classes"] = list(obj["classes"])
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            model = build_model(pretrained=False)
            model.load_state_dict(sd)
        state["model"] = model.eval()
        state["error"] = None
    except Exception as e:                                     # surface a readable message
        state["error"] = f"Could not load model: {e}"


@app.on_event("startup")
def _startup():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    load_model()


def gradcam(model, x, cls_idx):
    """Grad-CAM on the last convolutional block of MobileNetV3; returns HxW map in [0,1]."""
    feats, grads = {}, {}
    layer = model.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad()
        with torch.enable_grad():
            xx = x.clone().requires_grad_(True)
            model(xx)[0, cls_idx].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().numpy()
    finally:
        h1.remove(); h2.remove()


def overlay_png_b64(img: Image.Image, cam: np.ndarray) -> str:
    base = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    heat = np.stack([r, g, b], -1)
    out = (0.55 * base + 0.45 * heat).clip(0, 1)
    buf = io.BytesIO(); Image.fromarray((out * 255).astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"model_loaded": state["model"] is not None, "error": state["error"],
            "classes": state["classes"], "threshold": CONF_THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = False):
    if state["model"] is None:
        raise HTTPException(503, state["error"] or "Model not loaded.")
    data = await file.read()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(400, "Upload an image smaller than 8 MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not a readable image (use JPG or PNG).")

    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(state["model"](x), dim=1)[0]
    top = torch.topk(probs, k=min(3, len(state["classes"])))
    top3 = [{"label": state["classes"][i], "probability": round(float(p), 4)} for p, i in zip(top.values, top.indices)]
    label, conf = top3[0]["label"], top3[0]["probability"]
    auto = conf >= CONF_THRESHOLD
    res = {
        "label": label, "confidence": conf, "top3": top3, "threshold": CONF_THRESHOLD,
        "decision": "auto_sort" if auto else "manual_review",
        "stream": STREAMS.get(label, "Manual review") if auto else "Manual review required",
        "suggested_stream": STREAMS.get(label, "Manual review"),
        "message": ("Confident prediction: item can be routed automatically." if auto else
                    f"Confidence below {CONF_THRESHOLD:.2f}: send to a human for manual sorting."),
    }
    if explain:
        res["gradcam_png_base64"] = overlay_png_b64(img, gradcam(state["model"], x, int(top.indices[0])))
    return res


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
