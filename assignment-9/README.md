# Lab 9 — Vision-based Cognitive Systems (UCS420)

`Lab9_Vision_Cognitive_Assistants.ipynb` covers both tasks:

- **Part A — Medicine label reading assistant:** label localisation, multi-variant OCR (Tesseract + optional EasyOCR), drug knowledge base, expiry and daily-dose reasoning, medicine-cabinet memory (duplicate ingredient / interaction alerts), Q&A, speech output. Evaluated on a generated ground-truth label set (clean / mild / hard photos); real photos via Roboflow BottleLabels or `data/real_labels/`.
- **Part B — Emotion-aware assistant:** Haar face detection + CNN trained on FER-2013 (auto-download via kagglehub `msambare/fer2013`), temporal smoothing, valence tracking, escalation policy, mood timeline, webcam / Colab camera / video modes.

## Run
1. Open in Google Colab, set Runtime -> GPU (T4).
2. Run all cells. Setup installs Tesseract and packages.
3. If kagglehub asks for login: `import kagglehub; kagglehub.login()`, or download FER-2013 from Kaggle and extract to `data/fer2013/`.

Local: `pip install -r requirements.txt`, install Tesseract, then run in Jupyter.

## Optional env vars
| Variable | Effect |
|---|---|
| `GEMINI_API_KEY` | LLM-generated responses (falls back to rules) |
| `ROBOFLOW_API_KEY` | Downloads real bottle-label photos |
| `LAB9_EPOCHS` | Override training epochs (default 40 GPU / 10 CPU) |
| `LAB9_SKIP_TRAINING=1` | Reuse `outputs/emotion_cnn_best.pt` |

## Expected results
- Part A (Tesseract only): ~98% brand/strength, ~92% expiry, ~95% batch overall; 100% on clean and mild photos.
- Part B (GPU, full FER-2013): ~66–70% test accuracy.
