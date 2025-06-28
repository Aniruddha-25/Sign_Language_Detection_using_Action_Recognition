import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Parameters
batch_size = 128
imageSize = 96
num_classes = 29
train_dir = r'd:\Project\Sign Language Detection using Action Recognition\data\asl_alphabet_train\asl_alphabet_train'

label_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Data generators with augmentation
datagen = ImageDataGenerator(
    validation_split=0.15,
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.6,1.4],
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(imageSize, imageSize),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=label_map,
    seed=42
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(imageSize, imageSize),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=label_map,
    seed=42
)

# Build MobileNetV2-based model
base_model = MobileNetV2(
    input_shape=(imageSize, imageSize, 3),
    include_top=False,
    weights='imagenet',
    alpha=1.0
)
base_model.trainable = False

inputs = Input(shape=(imageSize, imageSize, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

steps_per_epoch = max(1, train_gen.samples // (batch_size * 2))
validation_steps = max(1, val_gen.samples // (batch_size * 2))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Initial training
history = model.fit(
    train_gen,
    epochs=12,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stop, lr_scheduler],
    verbose=2,
)

# Fine-tune last 80 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history_finetune = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stop, lr_scheduler],
    verbose=2,
)

# Save trained model
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, "ASL.h5")
model.save(model_save_path)
print(f"Model saved successfully at {model_save_path}")

# Plot training history
metrics = pd.DataFrame(history.history)
metrics[["loss", "val_loss"]].plot()
plt.title("Loss")
plt.show()
metrics[["accuracy", "val_accuracy"]].plot()
plt.title("Accuracy")
plt.show()

# Evaluate and show confusion matrix
val_gen.reset()
Y_pred = model.predict(val_gen, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_gen.classes
print(classification_report(y_true, y_pred, target_names=label_map))
plt.figure(figsize=(12, 12))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', xticklabels=label_map, yticklabels=label_map)
plt.title("Confusion Matrix")
plt.show()