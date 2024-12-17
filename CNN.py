import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Fonction pour prétraiter les images
def preprocess_images(images):
    processed_images = []
    for image in images:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image
        else:
            raise ValueError("Unsupported image format")
        resized_image = cv2.resize(gray_image, (100, 100))
        processed_image = np.array(resized_image) / 255.0
        processed_images.append(processed_image)
    processed_images = np.array(processed_images)
    return processed_images.reshape(-1, 100, 100, 1)

# Chemin du dossier contenant les images
chemin_dossier_images = "archive"

# Liste pour stocker les images
images = []
labels = []

# Parcourir tous les fichiers dans le dossier
for nom_fichier in os.listdir(chemin_dossier_images):
    # Vérifier si le fichier est une image .jpg
    if nom_fichier.endswith(".jpg"):
        # Chemin complet de l'image
        chemin_image = os.path.join(chemin_dossier_images, nom_fichier)
        # Charger l'image
        image = cv2.imread(chemin_image)
        # Vérifier si l'image est chargée avec succès
        if image is not None:
            images.append(image)
            # Ajouter l'étiquette (nom de fichier sans extension) aux labels
            labels.append(os.path.splitext(nom_fichier)[0])

# Prétraiter les images
processed_images = preprocess_images(images)

# Convertir les étiquettes en entiers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Création du modèle CNN
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Utilisation de softmax pour la classification multi-classe
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Définir la forme de l'entrée du modèle CNN
input_shape = (100, 100, 1)
num_classes = len(np.unique(labels_encoded))  # Nombre de classes

# Créer le modèle CNN
cnn_model = create_cnn_model(input_shape, num_classes)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(processed_images, labels_encoded, test_size=0.3, random_state=42)

# Entraîner le modèle
cnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Évaluer le modèle
train_loss, train_acc = cnn_model.evaluate(X_train, y_train)
val_loss, val_acc = cnn_model.evaluate(X_val, y_val)
print("Training Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)