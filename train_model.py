import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ============================
# Dataset Paths
# ============================
base_dir = r"C:\Users\iamar\Downloads\fruit-classifier"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# We’ll split test into validation + test
val_size = 0.5

# ============================
# Parameters
# ============================
img_size = (150, 150)
batch_size = 32

# Load datasets
raw_train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

raw_val_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    validation_split=val_size,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

raw_test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    validation_split=val_size,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Class names
class_names = raw_train_ds.class_names
print("Classes:", class_names)

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.prefetch(AUTOTUNE)
val_ds = raw_val_ds.prefetch(AUTOTUNE)
test_ds = raw_test_ds.prefetch(AUTOTUNE)

# ============================
# Data Augmentation
# ============================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ============================
# Transfer Learning Model
# ============================
base_model = keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze the convolutional base

model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================
# Train
# ============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# ============================
# Fine-tuning (unfreeze base_model)
# ============================
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ============================
# Evaluate
# ============================
loss, acc = model.evaluate(test_ds)
print(f"✅ Test accuracy: {acc:.2%}")

# ============================
# Save Model
# ============================
model.save("fruit_classifier_mobilenet.keras")
print("🎉 Model saved as fruit_classifier_mobilenet.keras")
