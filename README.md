# Deepfake Forensics — Detection & Analysis

An interactive web application that integrates a PyTorch-based deep learning model to classify images as "Real" or "Fake" and provides explainable AI insights through Grad-CAM visual heatmaps.

## Features
- **PyTorch Backend:** Utilizes a trained ResNet18 model for accurate deepfake detection.
- **Explainable AI (XAI):** Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps to highlight the areas of the image the model focused on to make its decision.
- **Interactive UI:** A sleek, user-friendly HTML/JS frontend that allows drag-and-drop image uploads and beautifully renders the analysis results.

## Technologies Used
- **Frontend**: HTML5, Vanilla JavaScript, CSS3
- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: PyTorch, TorchVision (ResNet18)
- **Computer Vision**: OpenCV, Pillow (PIL)
- **Explainable AI (XAI)**: Grad-CAM (`pytorch-grad-cam`)

## Project Structure
- `app.py`: The Python Flask backend server that handles image preprocessing, PyTorch model inference, and Grad-CAM generation.
- `Deepfake_Detection_Interactive_Demo.html`: The interactive HTML frontend.
- `Deepfake_Detection_Upgraded_Version_.ipynb`: The Jupyter Notebook used to train the ResNet18 model.
- `requirements.txt`: Python dependencies required to run the backend server.
- `resnet18_deepfake_model.pth`: The trained PyTorch model weights **(You must provide this file)**.

## Installation & Setup

1. **Prerequisites:** Ensure you have Python installed on your system.
2. **Install Dependencies:** Open a terminal in the project directory and install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Provide the Model Weights:** 
   Place your trained PyTorch model file named `resnet18_deepfake_model.pth` directly into this project directory. 
   *(Note: If the file is missing, the backend will automatically initialize an untrained, "dummy" model so you can still test the UI flow).*

## Usage

1. **Start the Backend Server:**
   Run the Flask server using Python:
   ```bash
   python app.py
   ```
   *(On Windows, you may need to use `py app.py`)*
   The server will start on `http://localhost:5000`.

2. **Open the Interactive Demo:**
   Simply double-click the `Deepfake_Detection_Interactive_Demo.html` file to open it in your web browser.

3. **Analyze an Image:**
   Drag and drop an image into the dropzone or click to upload. The UI will communicate with the backend and display the verdict, confidence score, and the Grad-CAM heatmap.
