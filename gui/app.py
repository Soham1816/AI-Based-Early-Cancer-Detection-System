import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('../model/cancer_model.h5')

with open('../model/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]


window = tk.Tk()
window.title("AI Cancer Detection")
window.geometry("500x650")
window.configure(bg="#f4f4f4")

logo_img = Image.open("college.png")  
logo_img = logo_img.resize((500, 100))
logo_photo = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(window, image=logo_photo, bg="#f4f4f4")
logo_label.image = logo_photo
logo_label.pack(pady=10)


title_label = tk.Label(window, text="AI Cancer Detection", font=("Helvetica", 20, "bold"), bg="#f4f4f4", fg="#222")
title_label.pack(pady=15)


img_label = tk.Label(window, bg="#f4f4f4")
img_label.pack(pady=10)


result_label = tk.Label(window, text="", font=("Helvetica", 14), bg="#f4f4f4", fg="#333")
result_label.pack(pady=10)

def predict_image(file_path):
    try:
        img = Image.open(file_path).convert("RGB").resize((224, 224))
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        pred_index = np.argmax(prediction)
        raw_class = class_names[pred_index]
        confidence = round(float(prediction[pred_index]) * 100, 2)

    
        parts = raw_class.split('_')
        if len(parts) == 2:
            organ = parts[0].capitalize()
            subtype = parts[1].replace('non', 'Non-').upper() if 'non' in parts[1] else parts[1].capitalize()
            formatted_class = f"{organ} – {subtype}"
        else:
            formatted_class = raw_class.capitalize()

        result_label.config(
            text=f"Prediction: {formatted_class}\nConfidence: {confidence}%",
            fg="#007700"
        )

    except Exception as e:
        result_label.config(text="Error: " + str(e), fg="red")


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        predict_image(file_path)

upload_btn = tk.Button(window, text="Upload Image", command=upload_image, font=("Helvetica", 12), width=20)
upload_btn.pack(pady=20)


footer_note = tk.Label(window, text="Made By Soham Adivarekar SE-07", font=("Helvetica", 10, "italic"), bg="#f4f4f4", fg="#777")
footer_note.pack(side="bottom", pady=15)

window.mainloop()
