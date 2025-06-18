import os
from pathlib import Path
import shutil

classes = [
    "Tomato", "Carrot", "Cucumber", "Potato", "Radish"
]

# Create mapping: class name → ID
class_to_id = {cls: i for i, cls in enumerate(classes)}

# Save class names to file (optional)
with open("classes.txt", "w") as f:
    print('loop1')
    for cls in classes:
        print('loop2')
        f.write(f"{cls}\n")

# For each data split
for split in ['train', 'validation', 'test']:
    print('loop3')
    original_dir = Path(f"data/{split}")
    image_dir = original_dir / "images"
    label_dir = original_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in original_dir.iterdir():
        print(f'🧺 Found class folder: {class_folder.name}')
        if class_folder.is_dir() and class_folder.name not in ['images', 'labels']:
            class_name = class_folder.name
            class_id = class_to_id.get(class_name)
            if class_id is None:
                print(f"⚠️ Unknown class: {class_name}, skipping...")
                continue

            count = 0
            for img_file in class_folder.glob("*.[jp][pn]g"):
                new_img_path = image_dir / img_file.name
                shutil.copy(img_file, new_img_path)

                label_file = label_dir / (img_file.stem + ".txt")
                with open(label_file, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                count += 1

            print(f"✅ Processed {count} images in {class_name}")