# Emotion Recognition with Neural Networks

This project builds a neural network to classify human emotions based on facial images. The model is trained on preprocessed datasets and deployed in real-time to detect and classify emotions through a camera feed.

---

## Project Structure

```
cropped_faces/
|
|-- test/              # Test data (organized by emotion category)
|   |-- angry/
|   |-- disgust/
|   |-- fear/
|   |-- happy/
|   |-- neutral/
|   |-- sad/
|   |-- surprise/
|
|-- train/             # Training data (organized by emotion category)
|   |-- angry/
|   |-- disgust/
|   |-- fear/
|   |-- happy/
|   |-- neutral/
|   |-- sad/
|   |-- surprise/
|
|-- Face.ipynb         # Jupyter Notebook for data exploration and model training
|-- main.py            # Script for real-time emotion detection
|-- mymodel.pkl        # Pretrained model saved as a pickle file
|-- README.md          # Documentation for the project
```

---

## Features

1. **Data Preprocessing**:
   - Facial images are categorized into seven emotion labels: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
   - Images are preprocessed and organized into `train` and `test` folders for training and evaluation.

2. **Model Training**:
   - The neural network is trained using the `train` dataset.
   - The model is saved as `mymodel.pkl` for later use.

3. **Real-Time Detection**:
   - The `main.py` script uses the pretrained model to detect emotions in real-time.
   - A webcam captures live video, and the model classifies the emotion displayed on the detected face.

---

## Installation

### Prerequisites
- Python 3.x
- Virtual environment (recommended)

### Required Libraries
Install the dependencies using pip:
```bash
pip install -r requirements.txt
```
Create a `requirements.txt` file with the following contents:
```
numpy
opencv-python
scikit-learn
tensorflow
keras
matplotlib
```

---

## Usage

### Step 1: Train the Model
If you want to retrain the model, open `Face.ipynb` in Jupyter Notebook and run all cells. This will preprocess the data, train the model, and save it as `mymodel.pkl`.

### Step 2: Real-Time Emotion Detection
Run the `main.py` script to start the webcam and classify emotions in real-time:
```bash
python main.py
```

### Output
- The detected emotion category will be displayed in real-time on the video feed.
- Example categories: `Happy`, `Sad`, `Angry`, etc.

---

## Model Details
The neural network was designed with the following architecture:
- Input Layer: Preprocessed facial images.
- Hidden Layers: Fully connected layers with ReLU activation.
- Output Layer: Softmax layer for emotion classification.

The model was optimized using categorical cross-entropy loss and Adam optimizer.

---

## Dataset
The dataset is divided into:
1. **Train Dataset**: Used to train the model.
2. **Test Dataset**: Used to evaluate the model's performance.

Each dataset folder contains subfolders corresponding to the emotion labels.

---

## Future Enhancements
- Improve accuracy by adding more training data.
- Integrate a preprocessing pipeline to handle different lighting and background conditions.
- Deploy the model on edge devices for wider applications.

---

