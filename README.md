🍏 Fruit Classifier
A deep learning project to classify fruits using TensorFlow and Keras. This project leverages transfer learning with MobileNetV2 and provides a straightforward workflow for training and real-time predictions.

🚀 Key Features
Advanced Classification: Utilizes a deep convolutional neural network for high-accuracy fruit classification.

Transfer Learning: Employs a pre-trained MobileNetV2 model to accelerate training and improve performance.

Real-time Prediction: Supports real-time image predictions with a simple command-line interface.

Comprehensive Dataset: Trained on the extensive Fruits-360 dataset.

<br>

<br>

📸 Project Demo
![Fruit Prediction Example](./assets/Screenshot 2025-09-07 205549.png)
Example prediction showing Onion Red 2 with 100% confidence.

🗂 Dataset
The project uses the Fruits-360 dataset, which contains over 100k images of fruits across 210 classes.

Training set: ~67,692 images

Test set: ~22,688 images

Image size: 150x150 pixels

<br>

Note: The dataset is automatically downloaded and structured when you run the training script, but you can also access it directly via the following link: Fruits-360

⚙️ Project Structure
The project is organized into a clean and easy-to-navigate directory structure.

fruit-classifier/
├── train/              # Training images
├── test/               # Test images
├── assets/             # Project assets like screenshots and diagrams
├── train_model.py      # Script to train the model
├── predict.py          # Script to predict a single image
├── fruit_classifier.keras # Trained model file
└── README.md



🛠 Model Training & Workflow
This section outlines the step-by-step process of training the model.

📝 Workflow
Data Preparation: The training script automatically loads and preprocesses images from the train/ and test/ directories.

Model Initialization: A pre-trained MobileNetV2 model is loaded as the base, with its layers initially frozen to preserve learned features.

Initial Training (15 epochs): Custom classification layers are added on top of the base model and trained on the dataset.

Fine-tuning (5 epochs): The base model's final layers are unfrozen and trained with a very low learning rate to fine-tune the model's performance on the fruit dataset.

Model Export: The final, fine-tuned model is saved as fruit_classifier.keras for future use.

📈 Training Metrics
Final Test Accuracy: ~98%

Trainable Parameters: ~2 million

Total Training Time: ~2 hours (dependent on hardware)

🖼 Image Prediction
The predict.py script makes it simple to classify a new image using the trained model.

💻 Usage
python predict.py "test/Lemon Meyer 1/28_100.jpg"



Output Example:

Predicted Class: Onion Red 2 (Confidence: 100.00%)



⚠️ Challenges & Solutions
Large Dataset Size: The Fruits-360 dataset is large, which can lead to extended training times. This was mitigated by using transfer learning, which significantly reduced the time needed to achieve high accuracy.

TensorFlow Warnings: TensorFlow often displays warnings about missing CPU instructions (SSE, AVX, FMA). These are informational and can be safely ignored.

File Path Management: Handling file paths, especially with spaces, required careful string formatting in scripts to ensure cross-platform compatibility.

📈 Future Improvements
Alternative Architectures: Explore other pre-trained models like ResNet or EfficientNet for potential performance gains.

Hyperparameter Tuning: Fine-tune the learning rate, dropout rate, and batch size to optimize model performance.

User Interface: Develop a simple GUI or web application to allow users to upload images and get instant predictions.

📌 Requirements
Dependencies:

Python 3.11

TensorFlow 2.x

NumPy

Pillow

Installation:

pip install tensorflow numpy pillow


