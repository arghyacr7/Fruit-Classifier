import os
import shutil

# Base dataset path
base_dir = r"C:\Users\iamar\Downloads\fruit-classifier"

# Which folders to clean
folders_to_fix = ["training", "test"]

for folder in folders_to_fix:
    loose_images_dir = os.path.join(base_dir, folder)

    print(f"\n🔎 Organizing folder: {loose_images_dir}")

    for file in os.listdir(loose_images_dir):
        file_path = os.path.join(loose_images_dir, file)

        if os.path.isdir(file_path):
            continue  # skip already organized subfolders

        name, ext = os.path.splitext(file)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue  # skip non-images

        # Handle multi-fruit images (apple_grape, cherry_strawberries, etc.)
        if "_" in name or "(" in name:
            class_name = "Mixed"
        else:
            # Capitalize properly (apple → Apple, mango → Mango)
            class_name = name.split()[0].capitalize()

        target_folder = os.path.join(loose_images_dir, class_name)
        os.makedirs(target_folder, exist_ok=True)

        shutil.move(file_path, os.path.join(target_folder, file))
        print(f"✅ Moved {file} → {class_name}/")

print("\n🎉 Finished organizing Training/ and Test/ folders!")
