# DeepFake-Detection-Model🔍
A deep learning-based DeepFake Image Detection System that classifies images or videos as Real or Fake using an EfficientNetB0 model. The project includes image preprocessing, model training, fine-tuning, evaluation, and a Streamlit web application for testing uploaded data.

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

## 🎯 Objective

The main objective of this project is to develop an automated system capable of identifying manipulated or AI-generated images and distinguishing them from genuine images.

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


## Screenshots
<img width="1033" height="820" alt="image" src="https://github.com/user-attachments/assets/75be3479-414c-41e6-aaef-afab799ef321" />
<img width="1074" height="806" alt="2" src="https://github.com/user-attachments/assets/779219b6-f486-4955-981f-991b8394248c" />
<img width="1030" height="793" alt="3" src="https://github.com/user-attachments/assets/cffb1079-dcd3-491f-854a-92843edd8141" />



## ⚠️ Disclaimer

This project is developed for **educational and research purposes**. DeepFake detection models may not be perfectly accurate, especially when dealing with unseen manipulation techniques or high-quality synthetic images.

