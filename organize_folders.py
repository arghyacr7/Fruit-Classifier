import os
import shutil

base_dir = r"C:\Users\iamar\Downloads\fruit-classifier"

# Correct lowercase paths
train_dir = os.path.join(base_dir, "training")
test_dir = os.path.join(base_dir, "test")

# Make sure training/ and test/ exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Move fruit folders into training/ (ignore training and test themselves)
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)

    if os.path.isdir(item_path) and item not in ["training", "test", "papers", "test-multiple_fruits"]:
        print(f"📦 Moving {item} → training/")
        shutil.move(item_path, train_dir)

print("✅ All fruit folders moved into training/")
