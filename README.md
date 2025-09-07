# 🍎🥕🍊 Fruits Detection with YOLOv11s and Luxonis OAK Cameras

This repository contains an **educational project** for training and deploying a custom object detection model with **YOLOv11 (small variant)** to detect **apples, carrots, and oranges**.

---

## 📖 Overview

- **Training**: A YOLOv11s model trained on a compact dataset of fruits.  
- **Inference**: Scripts to run detection on both **PC** (with webcam) and **Luxonis OAK devices**.  
- **Evaluation**: Curves, confusion matrix, and metrics included in the `results/` folder.  
- **Purpose**: Educational and open-source, showing end-to-end workflow from dataset → training → deployment.  

---

## 📂 Repository Contents

- `load_oak_cam.py` → Python script to run the trained model on **OAK-1 Lite** (also compatible with OAK-D Lite and other Luxonis devices powered by the Intel Movidius Myriad VPU).  
- `results/` → Precision-recall curves, F1 curve, confusion matrix, and sample outputs.  
- Training and splitting scripts (adapted from Evan Juras, see Credits).  
- Links to dataset and models hosted on Hugging Face.  
---

## 📊 Training Results

- **Framework**: Ultralytics YOLOv11s  
- **Input size**: 640×640  
- **Epochs**: 60  
- **Dataset size**: 160 images (apples, carrots, oranges)  
- **Metrics**:  
  - mAP@50: **0.945**  
  - mAP@50–95: **0.920**  
---

## 🧾 Dataset

👉 [Fruits Dataset (Hugging Face)](https://huggingface.co/datasets/johnatanvq/fruits-dataset/tree/main)

- 160 images captured in different **angles, distances, lighting conditions, shadows, and surfaces**.  
- Labeled with **Label Studio** in YOLO format.  
---

## 🧠 Models

👉 [Fruits YOLOv11 Models (Hugging Face)](https://huggingface.co/johnatanvq/fruits-yolo-model/tree/main)
- `my_model.pt`: PyTorch weights for PC/server inference.  
- `my_model_openvino_2022.1_6shave.blob`: Optimized model for **OAK devices**.  
  - Converted via **Luxonis third-party service**.  
  - Includes `.onnx`, `.bin`, `.xml`, `.json` as part of the conversion output.  

**Why two models?**  
- `.pt` → for training and inference on PC.  
- `.blob` → required by Luxonis DepthAI for deployment on OAK hardware.  

---
## ⚠️ On OAK-1 Lite, inference currently runs at ~4-5 FPS.

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/ce639ee0-cf71-41be-87ac-e1c0893608be" alt="oak_cam" width="49%">
  <img src="https://github.com/user-attachments/assets/8b7d081c-7659-47b4-838a-e4913d2cb022" alt="oak_video" width="49%">
</div>

Possible causes:

<ul>
  <li>Model too heavy for the hardware.</li>
  <li>Script configuration (load_oak_cam.py) could be optimized.</li>
  <li>High input resolution (trained at 640×640).</li>
</ul>

Improvements possible by adjusting resolution, thresholds, pruning, or quantization.

## 📜 License

Code & Models (this repo): Apache License 2.0

Dataset: CC-BY 4.0

You are free to use, share, and adapt the project — credit to the author is appreciated.

## 🙏 Credits

This project builds upon the excellent work of **Evan Juras**:

[Training Notebook](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=PooP5Vjsg2Jn)

[YouTube Tutorial](https://www.youtube.com/watch?v=r0RspiLG260)

➡️ Adapted and updated for YOLOv11 training and deployment on Luxonis OAK cameras.

## 📝 Notes

Developed for educational purposes.

Compact dataset but strong performance (mAP@50 ~0.95).

results/ contains evaluation outputs (curves, confusion matrix, samples).

Open for improvements: feel free to experiment with input sizes, anchors, and hyperparameters.
