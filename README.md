# DeepFake-Detection-Model🔍
A deep learning-based DeepFake Image Detection System that classifies images or videos as Real or Fake using an EfficientNetB0 model. The project includes image preprocessing, model training, fine-tuning, evaluation, and a Streamlit web application for testing uploaded data.

# Problem Statement –
The rapid advancement of Artificial Intelligence and Generative AI has made it increasingly easy to create realistic deepfake images by manipulating or generating human faces and other visual content. These fake images can be used to spread misinformation, damage reputations, commit fraud, and deceive individuals.

Traditional methods of identifying manipulated images through manual inspection are often unreliable because modern deepfakes can appear highly realistic and contain subtle modifications that are difficult for humans to detect.

Therefore, there is a need for an automated Deepfake Detection System that can analyze an input image and determine whether it is Real or Fake. The proposed system uses Deep Learning and Computer Vision techniques, particularly a convolutional neural network-based model, to learn visual patterns and artifacts associated with manipulated images.

The system aims to provide accurate and reliable classification of images as Real or Fake, along with a prediction confidence score. It can be integrated into a user-friendly application where users can upload an image and receive the detection result.

## 🎯 Objective
The main objective of this project is to develop an automated system capable of identifying manipulated or AI-generated images and distinguishing them from genuine images.
Apply deep learning techniques to extract meaningful visual features.
Use EfficientNetB0 for image classification.
Prepare and process images in a consistent format.
Evaluate performance using classification metrics.
Build a simple Streamlit interface for uploading and testing images.
Identify model limitations and potential improvements.
## 🚀 Features

* 🔍 Detects whether an image/video is **Real or Fake**
* 🧠 Uses **EfficientNetB0** for deep learning-based classification
* 🖼️ Supports image upload for real-time prediction
* 🔄 Image preprocessing and augmentation
* 📊 Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
* 🎯 Fine-tuning to improve Fake image detection
* 🌐 Interactive **Streamlit** web interface
* 🐍 Built using Python and TensorFlow/Keras

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* EfficientNetB0
* OpenCV
* NumPy
* Pillow
* Streamlit
* Matplotlib
* Seaborn

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The system particularly focuses on improving **Fake-class recall**, helping reduce the number of fake images incorrectly classified as real.

## 🔄 Complete Project Workflow
                 DATASET
                    │
                    ▼
          Fake Images + Real Images
                    │
                    ▼
             Image Loading
                    │
                    ▼
          Image Preprocessing
             Resize 150×150
                    │
                    ▼
             Dataset Shuffling
                    │
                    ▼
            Train/Test Split
                    │
                    ▼
             Label Encoding
          Fake = 0 | Real = 1
                    │
                    ▼
             EfficientNetB0
          ImageNet Transfer Learning
                    │
                    ▼
       Global Average Pooling
                    │
                    ▼
                Dropout
                    │
                    ▼
          Dense Classification
             Fake / Real
                    │
                    ▼
               Training
                    │
                    ▼
             Trained Model
             my_model.keras
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Model Evaluation       Streamlit App
          │                   │
          ▼                   ▼
 Accuracy / Precision    Upload Image/Media
 Recall / F1-Score             │
 Confusion Matrix               ▼
                          Model Prediction
                                │
                                ▼
                         FAKE or REAL

## ✨ Key Features

Binary image classification: Predicts REAL or FAKE labels.
Deep learning: Uses an EfficientNetB0-based image classification model.
Image preprocessing: Resizes images to the model's expected input dimensions.
Transfer learning: Reuses pretrained features if enabled in the final training configuration.
Model evaluation: Supports analysis using accuracy, precision, recall, F1-score, and a confusion matrix.
Interactive interface: Allows users to upload images through Streamlit.
Modular implementation: Separates data loading, model training, and application inference.

## ⚙️ How It Works

The system follows these steps:
1. Image upload: The user selects an image through the application.
2. Preprocessing: The image is loaded and transformed into the input format expected by the model.
3. Feature extraction: EfficientNetB0 processes the image and extracts learned visual representations.
4. Classification: The classification head generates an output for the REAL and FAKE categories.
5. Prediction display: The application presents the predicted class and, if implemented, a confidence score.

## 💻 Running the Project

Clone the repository and install the required dependencies:

```bash
git clone <https://github.com/choprapurvi168-hub/DeepFake-Detection-Model>

1. Install Dependencies

pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow streamlit

2. Train / Load Model

Open code.ipynb
Run the notebook to preprocess data, train EfficientNetB0, evaluate and save:
my_model.keras

3. Launch Web App

streamlit run app.py

4. Test the Model

Open the Streamlit link in browser
Upload an image
Image → Preprocessing → my_model.keras → FAKE / REAL


## 📂 Dataset

The project uses two image categories:

1. REAL: Images labelled as authentic in the selected dataset.
2. FAKE: Images labelled as synthetic or manipulated in the selected dataset.

The images are organized into labelled classes so that the model can learn from examples.
