import os

# List of target class folders
folders = [
    "skin_benign",
    "skin_malignant",
    "lung_normal",
    "lung_cancer",
    "breast_idc",
    "breast_non_idc"
]

# Root dataset path
root = "datasets"

# Create each folder
for folder in folders:
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created folder: {path}")

print("\n🎯 All dataset folders are ready.")
