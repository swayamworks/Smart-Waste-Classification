# Smart Waste Classification using Computer Vision

Course: Deep Learning (ADCOR1PC301) · TYAIDS · Tanmay Badhe (Roll No: 3) & Swayam Netke (Roll No: 33)

A simple end-to-end version of the microproject: **model → backend → frontend**.

| Part | File | What it does |
|---|---|---|
| Model | `train.py` | Trains MobileNetV3-Large on TrashNet (six classes) using the Phase 2 recipe |
| Shared settings | `app/common.py` | Classes, 224×224 preprocessing, 0.90 threshold, sorting streams |
| Backend | `app/main.py` | FastAPI: `/predict`, `/health`, serves the frontend |
| Frontend | `frontend/index.html` | Upload / camera page showing class, confidence, stream, top-3, Grad-CAM |

## 1. Install
```bash
pip install -r requirements.txt
```

## 2. Get the model (pick one)
**A. Use the checkpoint you already trained** (the `.pt` inside `smart_waste_results.zip`):
copy it to `models/mobilenetv3_large_trashnet.pt`. A plain `state_dict` or a dict with
`model_state` / `state_dict` works. Class order must be: cardboard, glass, metal, paper, plastic, trash.

**B. Train again** (GPU recommended, e.g. Colab T4):
```bash
# download TrashNet "dataset-resized" from https://github.com/garythung/trashnet and unzip it
python train.py --data path/to/dataset-resized
```
This audits the data (removes conflicting duplicates), groups near-identical photos, splits 70/15/15
at group level (seed 42), trains with warm-up + fine-tuning, restores the best validation macro-F1
checkpoint, prints test accuracy / macro precision / recall / F1 / confusion matrix and saves
`models/mobilenetv3_large_trashnet.pt` plus a `.metrics.json`.

## 3. Run
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 · API docs at http://127.0.0.1:8000/docs

## API
`POST /predict?explain=false` (multipart field `file`) returns:
```json
{"label":"glass","confidence":0.97,"decision":"auto_sort","stream":"Dry: glass",
 "suggested_stream":"Dry: glass","top3":[{"label":"glass","probability":0.97}, ...],
 "threshold":0.9,"message":"..."}
```
If confidence < 0.90, `decision` is `manual_review` (the human-review fallback from Phase 3).
`explain=true` adds `gradcam_png_base64`.

## Honest limitations (same as the report)
- Six classes only; no organic, sanitary, hazardous, battery or e-waste.
- Trained on TrashNet (one clear item per photo); not validated on a real smart-bin or cluttered waste.
- At the 0.90 threshold only about 28% of TrashNet test images were auto-accepted in Phase 3; the rest go to manual review.
- Do not use the INT8 ONNX export (accuracy fell to 51.6%); this app serves the FP32 PyTorch model.
