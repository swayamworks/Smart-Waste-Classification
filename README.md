# Smart Waste Classification using Computer Vision

**Course**: Deep Learning (ADCOR1PC301)  
**Class**: TYAIDS  
**Team**:
- **Tanmay Badhe** — Roll No: 3 (PRN: 124BTAI1026)
- **Swayam Netke** — Roll No: 33 (PRN: 124BTAI1066)

---

## 🚀 1-Click Cloud Deployment

Deploy the live full-stack app (FastAPI + PyTorch MobileNetV3-Large + Frontend UI) to the cloud for free with one click:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/swayamworks/Smart-Waste-Classification)

---

## 📁 Repository Contents

- **`TYAIDS-3,33_SLA_PHASE-1.docx`**: Phase 1 Report (Problem Formulation & Dataset Audit).
- **`TYAIDS-3,33_SLA_PHASE-2.docx`**: Phase 2 Report (Architecture Benchmarking & Two-Stage Transfer Learning).
- **`TYAIDS-3,33_SLA_PHASE-3.docx`**: Phase 3 Report (Selective Classification, Reject Option & Model Auditing).
- **`Smart_Waste_Classification.ipynb`**: Complete end-to-end Google Colab / Jupyter notebook.
- **`smart_waste_app/`**:
  - `models/mobilenetv3_large_trashnet.pt`: Trained FP32 PyTorch checkpoint (16.2 MB).
  - `app/main.py`: FastAPI backend with `/predict`, `/health`, and Grad-CAM explainability.
  - `frontend/index.html`: Responsive web frontend with live camera/upload, confidence scores, and stream routing.
  - `train.py`: Self-contained training script.

---

## 💻 Local Quickstart

```bash
cd smart_waste_app
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
