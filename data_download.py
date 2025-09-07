import kagglehub
import os
import shutil

# ============================
# 1️⃣ Download Fruits-360 dataset
# ============================
dataset_path = kagglehub.dataset_download("moltean/fruits")
print("✅ Downloaded Fruits-360 dataset to:", dataset_path)

# ============================
# 2️⃣ Set source and destination paths
# ============================
src_dataset = os.path.join(dataset_path, "fruits-360_100x100", "fruits-360")
src_train = os.path.join(src_dataset, "Training")
src_test = os.path.join(src_dataset, "Test")

dst_base = r"C:\Users\iamar\Downloads\fruit-classifier"
dst_train = os.path.join(dst_base, "train")
dst_test = os.path.join(dst_base, "test")

# Create destination folders if they don't exist
os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_test, exist_ok=True)

# ============================
# 3️⃣ Function to copy images into class subfolders
# ============================
def copy_images(src_folder, dst_folder):
    print(f"Organizing images from {src_folder}...")
    classes = os.listdir(src_folder)
    for class_name in classes:
        src_class_folder = os.path.join(src_folder, class_name)
        dst_class_folder = os.path.join(dst_folder, class_name)
        os.makedirs(dst_class_folder, exist_ok=True)

        # Copy all images from source to destination class folder
        for file_name in os.listdir(src_class_folder):
            src_file = os.path.join(src_class_folder, file_name)
            dst_file = os.path.join(dst_class_folder, file_name)
            shutil.copy2(src_file, dst_file)
    print(f"✅ Finished organizing {dst_folder}")

# ============================
# 4️⃣ Copy train and test images
# ============================
copy_images(src_train, dst_train)
copy_images(src_test, dst_test)

print("🎉 Dataset ready! You can now train your model on 'train/' and test on 'test/'")
