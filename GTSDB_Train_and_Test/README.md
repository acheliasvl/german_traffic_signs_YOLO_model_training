# YOLOv8 German Traffic Sign Recognition (GTSDB)

This project implements a real-time traffic sign detection system using the **YOLOv8** architecture. It is specifically trained on the **German Traffic Sign Detection Benchmark (GTSDB)** to recognize 43 different classes of road signs via a live webcam feed.

---

## 🚀 Project Overview

The goal of this project is to provide a lightweight, efficient pipeline for training and deploying a traffic sign detector. 

* **Model Architecture:** YOLOv8 Nano (`yolov8n.pt`).
* **Reasoning:** The **Nano** version was selected specifically to ensure compatibility with diverse hardware and to maintain high FPS (frames per second) during real-time webcam inference on machines with limited GPU/CPU resources.
* **Dataset:** [German Traffic Sign Detection Benchmark (GTSDB)](https://www.kaggle.com/datasets/icebearogo/german-traffic-sign-detection-gtsdb-dataset).

---

## 🛠️ Installation & Setup

1.  **Clone the Repository** (or download the scripts).
2.  **Install Dependencies:**
    ```bash
    pip install ultralytics opencv-python
    ```
3.  **Prepare the Dataset:**
    * Download the dataset from Kaggle.
    * Update the `path` variable in `gtsdb.yaml` to point to your local dataset directory.

---

## 📂 File Structure

* **`train.py`**: The entry point for starting a fresh training session for 50 epochs.
* **`resume_training.py`**: A utility script to pick up training from the last saved checkpoint (`last.pt`) if the process was interrupted.
* **`webcam.py`**: The inference script that uses your computer's webcam to detect signs in real-time.
* **`gtsdb.yaml`**: The configuration file defining the dataset paths and the 43 traffic sign classes.
* **`traffic_sign_model.pt`**: The trained model weights. 50 epoches are completed. 

---

## 🚦 Usage

### 1. Training the Model
To start training the YOLOv8 Nano model on the GTSDB dataset:
```bash
python train.py
```

### 2. Resuming Training
If your training session crashes or is stopped manually:
```bash
python resume_training.py
```

### 3. Real-time Detection
Once you have a trained model (e.g., `traffic_sign_model.pt`), run the webcam script:
```bash
python webcam.py
```

---

## ⚠️ Important Notes on Confidence

In `webcam.py`, the confidence threshold is currently set to a very low level:
```python
results = model(frame, conf=0.10, verbose=False)
```

> **Note:** The **10% confidence level** is intentional for early-stage testing and to account for models that have not yet reached full convergence. 
> 
> **Recommendation:** As your training progresses and the model becomes more accurate, it is highly recommended to **increase the confidence threshold** (e.g., to `0.40` or `0.50`) to reduce "ghost" detections and false positives. If you want higher accuracy, training **higher epoches** (400-500) is recommended.

---

## 📊 Dataset Classes
The model is trained to recognize 43 distinct classes, ranging from speed limits (20km/h to 120km/h) and yield signs to specific maneuvers like "Roundabout mandatory" and "Keep right." For a full list of classes, refer to the `gtsdb.yaml` file.