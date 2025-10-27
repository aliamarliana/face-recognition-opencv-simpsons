# 🟨 Simpsons Face Recognition  
A computer vision project that uses Convolutional Neural Networks (CNNs) and OpenCV to recognize characters from *The Simpsons*. Built with TensorFlow, Caer, and Canaro — this project demonstrates end-to-end image preprocessing, model training, and character prediction.

---

## ⚙️ Overview  
This project builds a face recognition system capable of identifying *The Simpsons* characters using grayscale image data.  
It leverages:
- **OpenCV** for image processing  
- **Caer** for data preprocessing and normalization  
- **Canaro** for simplified deep learning model training with Keras  

🧠 The model uses a CNN architecture inspired by VGG, trained on the **[Simpsons Characters Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset)**.

---

## 🪜 Setup Guide  

### 1️⃣ Create and Activate Virtual Environment  
It’s recommended to isolate your dependencies in a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2️⃣ Install Dependencies
Once the environment is activated, install the required libraries:

```bash
pip install opencv-contrib-python caer canaro tensorflow matplotlib numpy
```
You can also save them in a requirements.txt file for easier setup.


### 3️⃣ Download the Dataset
The project uses the Simpsons Characters Dataset from Kaggle.


### 4️⃣ Run the Application
The entire workflow — from data loading and preprocessing to training and prediction — runs from a single file:

```bash
python simpsons_app.py
```
This script will:
- Load and preprocess image data
- Build and train a CNN on the top 10 most frequent characters
- Validate model accuracy
- Test on an unseen image and print the predicted character

---

## 🧠 Key Concepts Demonstrated
- Image preprocessing (grayscale conversion, resizing, normalization)
- CNN architecture design and training
- Data augmentation for improved accuracy
- Model validation and visualization with Matplotlib
- Prediction using a trained deep learning model

--- 

## 🎥 Code Reference
This project was **inspired by the tutorial by FreeCodeCamp**:  
🔗 [OpenCV Course - Full Tutorial with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o&t=11517s)
