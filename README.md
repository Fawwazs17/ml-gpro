# Real-Time Bahasa Isyarat Malaysia (BIM) Recognition System

**Course:** BICS 4340 / CSCI 4340 — Machine Learning  
**Lecturer:** Prof. Dr. Amelia Ritahani binti Ismail  
**Semester:** Semester II, 2025/2026  

**Group G:**
| No. | Name | Matric No |
|-----|------|-----------|
| 1 | Iman Zafri bin Halim Shah | 2414499 |
| 2 | Muhammad Aizam bin Ahmad Marzuki | 2410687 |
| 4 | Adam Fawwaz bin Sazalizam | 2416969 |
| 5 | Muhammad Iman Imtiyaz bin Mohd Noor Fauzi | 2415293 |

---

## Project Overview

This project develops a real-time recognition system for **Bahasa Isyarat Malaysia (BIM)** — Malaysia's native sign language used by 300,000+ Deaf and Hard-of-Hearing individuals. The system classifies BIM hand signs using a **hybrid YOLOv11 + MediaPipe** approach, achieving higher accuracy than any single model alone.

**Dataset:** [Malaysian Sign Language (MSL) Image Dataset](https://www.kaggle.com/datasets/pradeepisawasan/malaysian-sign-language-msl-image-dataset) — 9,586 images across 29 classes (A–Z alphabet + numbers 0, 1, 10).

---

## Methodology

### Novel Algorithm
**YOLOv11** (not covered in class) combined with **MediaPipe** 21-point hand keypoint tracking — a hybrid ensemble that fuses geometric hand structure with raw image appearance.

### Hybrid Architecture
```
Input Image
    ├── MediaPipe → 21 hand landmarks (63 features) → MLP → class probabilities
    └── YOLOv11-cls → raw image features → class probabilities
                        ↓
              Soft-voting ensemble (50/50)
                        ↓
                Final BIM class prediction
```

### Baseline Comparisons (in-class algorithms)
| Model | Input |
|-------|-------|
| SVM (RBF kernel) | MediaPipe keypoints |
| Random Forest | MediaPipe keypoints |
| KNN | MediaPipe keypoints |

---

## Project Structure

```
ml-gpro/
├── BIM_Recognition.ipynb   # Main notebook (run this first)
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Python dependencies
├── dataset/                # MSL image dataset (download from Kaggle)
│   └── Dataset_MSL/
│       └── Dataset_MSL/
│           ├── Alphabet_MSL/   # A–Z (26 classes)
│           └── Number_MSL/     # 0, 1, 10 (3 classes)
├── models/                 # Saved models (generated after running notebook)
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   ├── knn_model.pkl
│   ├── mlp_model.pt
│   ├── yolo11_bim/weights/best.pt
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── config.json
└── yolo_dataset/           # Auto-generated YOLO training structure
```

---

## Setup & Installation

### Requirements
- Python 3.9+
- Apple Silicon (M1/M2) or NVIDIA GPU recommended

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download dataset
1. Go to: https://www.kaggle.com/datasets/pradeepisawasan/malaysian-sign-language-msl-image-dataset
2. Download and extract into `dataset/`

---

## Running the Project

### Step 1 — Train all models
Open and run all cells in `BIM_Recognition.ipynb`:
```bash
jupyter notebook BIM_Recognition.ipynb
# or open in VSCode
```

**Expected training times on M1 Max:**
- MediaPipe keypoint extraction: ~5–10 min
- SVM / RF / KNN: ~1–3 min each
- YOLOv11 (50 epochs): ~10–15 min
- MLP: ~2 min

### Step 2 — Launch the deployment app
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## Notebook Sections

| Section | Description |
|---------|-------------|
| 1 | Setup & package installation |
| 2 | Dataset loading |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | MediaPipe hand keypoint extraction |
| 5 | Baseline models — SVM, Random Forest, KNN |
| 6 | YOLOv11 classification (novel algorithm) |
| 7 | Hybrid MLP + YOLOv11 ensemble |
| 8 | Parameter tuning (grid search, optimiser comparison) |
| 9 | Evaluation & model comparison |
| 10 | Save all models |

---

## Deployment App

The Streamlit app (`app.py`) allows anyone to upload a BIM hand sign image and get an instant prediction:

- **Input:** Upload any `.jpg` or `.png` image
- **Output:** Predicted BIM class, confidence score, top-3 predictions, annotated hand landmarks
- **Model:** Hybrid soft-vote ensemble (MLP keypoints + YOLOv11)

---

## Evaluation Metrics

All models are evaluated on a held-out 20% test set using:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Confusion matrix

---

## SDG Alignment

- **SDG 4** — Quality Education: Enables Deaf students to interact with learning content in real time
- **SDG 3** — Good Health and Well-Being: Improves communication between DHH patients and healthcare providers
- **SDG 10** — Reduced Inequalities: Closes the communication gap for Malaysia's Deaf community

---

## Declaration of AI Usage

AI tools (Claude) were used to assist with code scaffolding, debugging, and report structuring. All intellectual content, analysis, methodology decisions, and conclusions were reviewed and validated by the group members. No AI-generated content was used without human oversight and revision.
