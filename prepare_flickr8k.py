import os
import csv
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Paths
image_dir = "data/Images"
caption_file = "data/captions.txt"
save_path = "flickr8k_dataset"

# Read CSV-style file: image,caption
entries = []
with open(caption_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header_skipped = False
    for row in reader:
        if not header_skipped:
            header_skipped = True
            continue
        if len(row) != 2:
            continue
        image_file, caption = row
        image_path = os.path.join(image_dir, image_file.strip())
        if os.path.isfile(image_path):
            entries.append({
                "image_path": image_path,
                "caption": caption.strip()
            })

print(f"✅ Processed {len(entries)} valid image-caption pairs.")

if not entries:
    raise ValueError("❌ No valid image-caption pairs found. Check file paths and format.")

# Split into train/val
train_data, val_data = train_test_split(entries, test_size=0.1, random_state=42)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Save
dataset_dict.save_to_disk(save_path)
print(f"✅ Dataset saved to {save_path}")
