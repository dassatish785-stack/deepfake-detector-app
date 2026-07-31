<h1 align="center">Deepfake Forensics — Detection & Analysis</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white" alt="HTML5">
</div>

<br>

An advanced, interactive web application that integrates a PyTorch-based deep learning model to classify images as **"Real"** or **"Fake"**. Built for transparency, the system provides **Explainable AI (XAI)** insights through Grad-CAM visual heatmaps, showing exactly *where* the model looked to make its decision.

---

## 🌟 Key Features

- **Robust PyTorch Backend:** Utilizes a highly optimized ResNet18 model fine-tuned specifically for deepfake detection.
- **Explainable AI (XAI):** Automatically generates **Grad-CAM** (Gradient-weighted Class Activation Mapping) heatmaps. It overlays attention masks on the original image, proving that the model isn't just guessing.
- **Sleek Interactive UI:** A gorgeous, vanilla HTML/JS/CSS frontend featuring drag-and-drop uploads, animated scanlines, and real-time processing feedback.
- **Production-Ready:** Pre-configured for deployment on cloud platforms like Render, using `gunicorn` and CPU-optimized PyTorch libraries.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| **Frontend** | HTML5, Vanilla JavaScript, CSS3 |
| **Backend** | Python, Flask, Flask-CORS, Gunicorn |
| **Machine Learning** | PyTorch, TorchVision (ResNet18 architecture) |
| **Computer Vision** | OpenCV, Pillow (PIL) |
| **Explainable AI** | Grad-CAM (`pytorch-grad-cam`) |

## 📂 Project Structure

- `app.py`: The core Flask API. It handles image preprocessing, PyTorch inference, Grad-CAM generation, and serves the frontend.
- `index.html`: The interactive frontend user interface.
- `Deepfake_Detection_Upgraded_Version_.ipynb`: The original Jupyter Notebook used to train and validate the ResNet18 model.
- `requirements.txt`: Python dependencies (optimized for CPU/cloud deployment).
- `resnet18_deepfake_model.pth`: The trained PyTorch model weights **(Required for real predictions)**.

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Open a terminal in the project directory and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Add the Model Weights
Place your trained PyTorch model file named `resnet18_deepfake_model.pth` directly into the root project directory. 
> **Note:** If the file is missing, the backend will automatically initialize an untrained, "dummy" model so you can still test the UI flow without crashing!

### 4. Start the Server
Run the Flask server:
```bash
python app.py
```
*(On Windows, you may need to use `py app.py`)*

### 5. Access the Web App
Open your web browser and navigate to:
**[http://localhost:5000](http://localhost:5000)**

Simply drag and drop an image into the scanner UI, and the system will stream the image to the Python backend for instant analysis!

---

## ☁️ Deployment

This project is fully configured to be deployed on cloud providers like **Render** for free. For step-by-step instructions on deploying the full stack to the cloud, please refer to the included [`DEPLOYMENT.md`](./DEPLOYMENT.md) guide.
