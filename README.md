\# AI-Based Cancer Detection System 🧠🩺



This capstone project uses \*\*Convolutional Neural Networks (CNN)\*\* to detect cancer from image datasets (e.g., skin cancer). It includes a trained model, training script, and a user-friendly GUI to help users test cancer detection using image input.



---



\## 🔬 Project Features



\- 🧠 \*\*CNN-based model\*\* for cancer classification

\- 📂 Preprocessed dataset: benign vs malignant images

\- 🏗️ `train\_model.py` to train your own model

\- 🎯 `cancer\_model.h5`: Trained Keras model for instant use

\- 🖼 GUI built using Tkinter (`app.py`) for testing

\- 📊 Class labels loaded from `class\_names.txt`

\- 🛠️ Structured, clean project layout



---



\## 📁 Folder Structure

capestone\_project/

│

├── datasets/

│ └── skin\_malignant/ # Image dataset

│

├── model/

│ ├── cancer\_model.h5 # Trained model

│ └── class\_names.txt # Class label names

│

├── gui/

│ ├── app.py # Tkinter-based GUI

│ └── college.png # Logo/image used in GUI

│

├── train\_model.py # Model training script

├── fix\_structure.py # Helper for directory fix (if any)

└── README.md # Project description (this file)











\## 🧪 How to Run



\### 🔧 Install Dependencies



```bash

pip install tensorflow keras numpy pillow

🚀 Run GUI

python gui/app.py

You’ll be prompted to upload an image. The model will predict whether the input is benign or malignant



🏋️‍♂️ Train Your Own Model

python train\_model.py

Make sure your dataset is structured correctly and update paths as needed.



📌 Requirements

Python 3.7+



TensorFlow / Keras



Tkinter (built-in with Python)



PIL (for image processing)



