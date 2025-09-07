# 🍏 Fruit Classifier

A deep learning project for classifying fruits using **TensorFlow** and **Keras**. This project leverages **transfer learning** with MobileNetV2 and provides a straightforward workflow for training and real-time predictions.

---

## 🚀 Key Features

- **Advanced Classification:** Utilizes a deep convolutional neural network for high-accuracy fruit classification.  
- **Transfer Learning:** Employs a pre-trained MobileNetV2 model to accelerate training and improve performance.  
- **Real-time Prediction:** Supports real-time image predictions via a simple command-line interface.  
- **Comprehensive Dataset:** Trained on the extensive Fruits-360 dataset.

---

## 📸 Project Demo

![Fruit Prediction Example](./assets/Screenshot%202025-09-07%20205549.png)  
*Example prediction showing Onion Red 2 with 100% confidence.*

---

## 🗂 Dataset

This project uses the **Fruits-360 dataset**, containing over **100k images** of fruits across **210 classes**.

- **Training set:** ~67,692 images  
- **Test set:** ~22,688 images  
- **Image size:** 150x150 pixels  

> ⚠️ Note: The dataset is automatically downloaded and structured when running the training script. You can also access it directly via the [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits).

---

## ⚙️ Project Structure

```
fruit-classifier/
├── train/                # Training images
├── test/                 # Test images
├── assets/               # Project assets like screenshots
├── train_model.py        # Script to train the model
├── predict.py            # Script to predict a single image
├── fruit_classifier.keras # Trained model file
└── README.md
```

---

## 🛠 Model Training & Workflow

### 📝 Workflow

1. **Data Preparation:** Automatically loads and preprocesses images from `train/` and `test/` directories.  
2. **Model Initialization:** Loads a pre-trained MobileNetV2 model as the base, initially freezing its layers to preserve learned features.  
3. **Initial Training (15 epochs):** Adds custom classification layers on top of the base model and trains on the dataset.  
4. **Fine-tuning (5 epochs):** Unfreezes final layers of the base model and trains with a very low learning rate to optimize performance.  
5. **Model Export:** Saves the final fine-tuned model as `fruit_classifier.keras`.

---

### 📈 Training Metrics

- **Final Test Accuracy:** ~98%  
- **Trainable Parameters:** ~2 million  
- **Total Training Time:** ~2 hours (hardware dependent)  

---

## 🖼 Image Prediction

The `predict.py` script allows classifying a new image easily:

```bash
python predict.py "test/Lemon Meyer 1/28_100.jpg"
```

**Example Output:**

```
Predicted Class: Onion Red 2 (Confidence: 100.00%)
```

---

## ⚠️ Challenges & Solutions

- **Large Dataset Size:** Training on Fruits-360 can be time-consuming. Mitigated using transfer learning to reduce time while maintaining high accuracy.  
- **TensorFlow Warnings:** Informational warnings about missing CPU instructions (SSE, AVX, FMA) can be safely ignored.  
- **File Path Management:** Carefully handled file paths, especially with spaces, to ensure cross-platform compatibility.

---

## 📈 Future Improvements

- **Alternative Architectures:** Experiment with ResNet, EfficientNet, or other pre-trained models.  
- **Hyperparameter Tuning:** Optimize learning rate, dropout, and batch size for better performance.  
- **User Interface:** Develop a GUI or web app for instant image predictions.

---

## 📌 Requirements

- **Python 3.11**  
- **TensorFlow 2.x**  
- **NumPy**  
- **Pillow**

**Installation:**

```bash
pip install tensorflow numpy pillow
```

