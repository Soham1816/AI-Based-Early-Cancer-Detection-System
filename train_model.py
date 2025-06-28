import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Paths
dataset_path = 'datasets'
model_path = 'model/cancer_model.h5'
class_names_file = 'model/class_names.txt'

# Parameters
image_size = (224, 224)
batch_size = 32
epochs = 10
val_split = 0.2

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=val_split
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save class names
with open(class_names_file, 'w') as f:
    for cls in train_generator.class_indices:
        f.write(f"{cls}\n")
class_names = list(train_generator.class_indices.keys())

# Load MobileNetV2 base
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # Freeze base layers

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Save model
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

# Plot accuracy & loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Final evaluation
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n🎯 Final Validation Accuracy: {val_acc * 100:.2f}%")

# Confusion matrix and report
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\n🧪 Confusion Matrix:")
print(confusion_matrix(val_generator.classes, y_pred))

print("\n📋 Classification Report:")
print(classification_report(val_generator.classes, y_pred, target_names=class_names))
