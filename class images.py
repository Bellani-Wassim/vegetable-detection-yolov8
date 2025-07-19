import os

base_dir = 'data'
splits = ['train', 'test', 'validation']

with open('classes.txt', 'r') as f:
    class_names = [line.strip().lower() for line in f]

class_map = {name: idx for idx, name in enumerate(class_names)}

def extract_class_from_filename(filename):
    for name in class_names:
        if name in filename.lower():
            return class_map[name]
    return None  

for split in splits:
    label_dir = os.path.join(base_dir, split, 'labels')
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(label_dir, file)
            true_class_id = extract_class_from_filename(file)
            if true_class_id is None:
                print(f"Skipping {file}, no class match found.")
                continue

            # Read and replace class IDs
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = str(true_class_id)
                    new_lines.append(' '.join(parts))

            with open(file_path, 'w') as f:
                f.write('\n'.join(new_lines))
