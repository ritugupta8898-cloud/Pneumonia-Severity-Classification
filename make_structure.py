import os

folders = [
    "data/raw",
    "data/processed",
    "data/annotations",
    "data/splits",
    "src",
    "notebooks",
    "reports/curves",
    "reports/gradcam_samples",
    "saved_models"
]

files = {
    "src/preprocess.py": "",
    "src/severity_label_gen.py": "",
    "src/dataloader.py": "",
    "src/model.py": "",
    "src/train_binary.py": "",
    "src/train_severity.py": "",
    "src/evaluate.py": "",
    "src/gradcam.py": "",
    "src/predict.py": "",
    "notebooks/EDA.ipynb": "",
    "notebooks/Train-Binary.ipynb": "",
    "notebooks/Train-Severity.ipynb": "",
    "notebooks/GradCAM-Visualization.ipynb": "",
    "README.md": "# Pneumonia Severity Classification\n\n",
    "requirements.txt": ""
}

# Create folders
for f in folders:
    os.makedirs(f, exist_ok=True)
    print(f"[DIR ] {f}")

# Create files
for f, content in files.items():
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"[FILE] {f}")

print("\nProject structure created successfully.")
