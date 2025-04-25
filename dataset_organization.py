import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# CONFIG
images_folder = "images_all"
labels_folder = "labels_all"
output_base = "dataset"
classes = ["window"]
train_ratio = 0.8

# Create folders
for split in ["train", "val"]:
    os.makedirs(f"{output_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_base}/labels/{split}", exist_ok=True)

# Match images with labels
image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
image_files = [f for f in image_files if os.path.exists(os.path.join(labels_folder, Path(f).stem + ".txt"))]

# Split
train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

# Copy function
def copy_files(files, split):
    for img_file in files:
        name = Path(img_file).stem
        label_file = name + ".txt"

        shutil.copy2(os.path.join(images_folder, img_file), f"{output_base}/images/{split}/{img_file}")
        shutil.copy2(os.path.join(labels_folder, label_file), f"{output_base}/labels/{split}/{label_file}")

# Move the files
copy_files(train_files, "train")
copy_files(val_files, "val")

# Make data.yaml
yaml_content = f"""train: {output_base}/images/train
val: {output_base}/images/val

nc: 1
names: {classes}
"""

with open(os.path.join(output_base, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(" Dataset is organized")
