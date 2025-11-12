import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split

# Paramètres
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
IMG_SIZE = (224, 224)

from google.colab import drive
drive.mount('/content/drive')

# Chemins des datasets
DATASET_TRAIN_PATH = "/content/drive/MyDrive/marine_sounds"
DATASET_TEST_PATH = "/content/drive/MyDrive/marine_sounds_test"

# Fonction pour convertir WAV en Mel spectrogramme
def wav_to_melspectrogram(path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalisation entre 0 et 1
    log_mel_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
    # Redimensionnement en 224x224 pour VGG19
    img = tf.image.resize(log_mel_norm[..., np.newaxis], IMG_SIZE)
    img = np.repeat(img.numpy(), 3, axis=-1)
    return img

# Visualisation de la fonction wav_to_melspectrogram
audio_file_path = '/content/drive/MyDrive/marine_sounds/Striped_Dolphin/75003061.wav'

# Généreration du spectrogramme de Mel
mel_spectrogram = wav_to_melspectrogram(audio_file_path)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram[:,:,0], sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# Construction dataset d'entraînement
print("Chargement des données d'entraînement...")
X_train, y_train_labels = [], []
labels = sorted(os.listdir(DATASET_TRAIN_PATH))

for idx, label in enumerate(labels):
    class_path = os.path.join(DATASET_TRAIN_PATH, label)
    for file in os.listdir(class_path):
        if file.endswith(".wav"):
            try:
                img = wav_to_melspectrogram(os.path.join(class_path, file))
                X_train.append(img)
                y_train_labels.append(idx)
            except Exception as e:
                print(f"Erreur avec {file}: {e}")

X_train = np.array(X_train)
y_train_categorical = tf.keras.utils.to_categorical(y_train_labels, num_classes=len(labels))
y_train_labels = np.array(y_train_labels)

print(f"Shape dataset entraînement: {X_train.shape} {y_train_categorical.shape}")

# Construction dataset de test
print("\nChargement des données de test...")
X_test, y_test_labels = [], []

for idx, label in enumerate(labels):
    class_path = os.path.join(DATASET_TEST_PATH, label)
    if os.path.exists(class_path):
        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                try:
                    img = wav_to_melspectrogram(os.path.join(class_path, file))
                    X_test.append(img)
                    y_test_labels.append(idx)
                except Exception as e:
                    print(f"Erreur avec {file}: {e}")

X_test = np.array(X_test)
y_test_categorical = tf.keras.utils.to_categorical(y_test_labels, num_classes=len(labels))
y_test_labels = np.array(y_test_labels)

print(f"Shape dataset test: {X_test.shape} {y_test_categorical.shape}")

# Split entraînement/validation (80/20)
X_train, X_val, y_train, y_val, y_train_lab, y_val_lab = train_test_split(
    X_train, y_train_categorical, y_train_labels, test_size=0.2, random_state=42
)

# Normalisation
X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

print(f"\nDonnées finales:")
print(f"Entraînement: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# VGG-19 Transfer Learning
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(len(labels), activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Entraînement du modèle
print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=16,
                    verbose=1)

# Sauvegarde du modèle
model.save("/content/drive/MyDrive/marine_sounds14_vgg19.keras", save_format='keras')
print("\nModèle sauvegardé avec succès!")

model.save("/content/drive/MyDrive/marine_sounds14_vgg19.h5", save_format='h5')

# Sauvegarde des poids
model.save_weights("/content/drive/MyDrive/marine_sounds14_vgg19.weights.h5")

# Construction dataset de test
print("\nChargement des données de test...")
X_test, y_test_labels = [], []
test_labels = []

# Obtenir tous les labels disponibles dans le dossier test
available_test_labels = sorted([d for d in os.listdir(DATASET_TEST_PATH)
                              if os.path.isdir(os.path.join(DATASET_TEST_PATH, d))])

print(f"Labels disponibles dans le test: {available_test_labels}")
print(f"Labels d'entraînement: {labels}")

# Vérifier la correspondance des labels
missing_labels = set(labels) - set(available_test_labels)
if missing_labels:
    print(f"ATTENTION: Labels manquants dans le test: {missing_labels}")

for idx, label in enumerate(labels):
    class_path = os.path.join(DATASET_TEST_PATH, label)
    if os.path.exists(class_path):
        test_labels.append(label)
        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                try:
                    img = wav_to_melspectrogram(os.path.join(class_path, file))
                    X_test.append(img)
                    y_test_labels.append(idx)
                except Exception as e:
                    print(f"Erreur avec {file}: {e}")

X_test = np.array(X_test)
y_test_labels = np.array(y_test_labels)

print(f"Shape dataset test: {X_test.shape}")
print(f"Nombre d'échantillons de test: {len(y_test_labels)}")
print(f"Labels présents dans le test: {test_labels}")

# Vérifier s'il y a des données de test
if len(X_test) == 0:
    print("ATTENTION: Aucune donnée de test trouvée!")
    # Utiliser les données de validation pour le test
    X_test, y_test_labels = X_val, y_val_lab
    print("Utilisation des données de validation pour le test")
else:
    # Convertir en categorical en utilisant le nombre de classes d'origine
    y_test_categorical = tf.keras.utils.to_categorical(y_test_labels, num_classes=len(labels))

# Évaluation sur l'ensemble de TEST
print("\n=== ÉVALUATION SUR L'ENSEMBLE DE TEST ===")

if len(X_test) > 0:
    # Prédictions sur l'ensemble de test
    y_test_pred_proba = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)

    # Filtrer les labels pour n'inclure que ceux présents dans le test
    unique_test_labels = np.unique(y_test_labels)
    present_labels = [labels[i] for i in unique_test_labels]

    print(f"Labels uniques dans les prédictions: {unique_test_labels}")
    print(f"Labels correspondants: {present_labels}")

    # Taux de reconnaissance (accuracy)
    test_accuracy = accuracy_score(y_test_labels, y_test_pred)
    print(f"Taux de reconnaissance sur TEST: {test_accuracy:.4f}")

    # Rapport détaillé avec les bons paramètres
    print("\nRapport de classification sur TEST:")
    try:
        # Utiliser seulement les labels présents dans le test
        report = classification_report(y_test_labels, y_test_pred,
                                     labels=unique_test_labels,
                                     target_names=[labels[i] for i in unique_test_labels],
                                     zero_division=0,
                                     output_dict=False)
        print(report)
    except Exception as e:
        print(f"Erreur avec le rapport de classification: {e}")
        # Version simplifiée
        print("Rapport simplifié:")
        print(classification_report(y_test_labels, y_test_pred, zero_division=0))

    # Matrice de confusion pour le test
    print("Matrice de confusion (TEST):")
    cm_test = confusion_matrix(y_test_labels, y_test_pred)

    # Affichage visuel de la matrice de confusion
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=[labels[i] for i in unique_test_labels],
                yticklabels=[labels[i] for i in unique_test_labels])
    plt.title('Matrice de confusion - Ensemble de TEST')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Calcul des métriques globales sur le test
    precision_test = np.diag(cm_test) / np.sum(cm_test, axis=0)
    recall_test = np.diag(cm_test) / np.sum(cm_test, axis=1)
    f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

    # Remplacer les NaN par 0 (cas où division par 0)
    precision_test = np.nan_to_num(precision_test, nan=0.0)
    recall_test = np.nan_to_num(recall_test, nan=0.0)
    f1_score_test = np.nan_to_num(f1_score_test, nan=0.0)

    print("\n=== MÉTRIQUES GLOBALES SUR TEST ===")
    print(f"Précision moyenne: {np.mean(precision_test):.4f}")
    print(f"Rappel moyen: {np.mean(recall_test):.4f}")
    print(f"F1-Score moyen: {np.mean(f1_score_test):.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    # Affichage des métriques par classe sur le test
    print("\n=== MÉTRIQUES PAR CLASSE SUR TEST ===")
    for i, label_idx in enumerate(unique_test_labels):
        label_name = labels[label_idx]
        print(f"{label_name}:")
        print(f"  Précision: {precision_test[i]:.4f}")
        print(f"  Rappel: {recall_test[i]:.4f}")
        print(f"  F1-Score: {f1_score_test[i]:.4f}")
        print(f"  Échantillons: {np.sum(y_test_labels == label_idx)}")
        print()
else:
    print("Aucune donnée de test disponible pour l'évaluation")

# Évaluation sur la validation pour comparaison
print("\n=== ÉVALUATION SUR LA VALIDATION ===")
y_val_pred_proba = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)
val_accuracy = accuracy_score(y_val_lab, y_val_pred)

print(f"Taux de reconnaissance sur VALIDATION: {val_accuracy:.4f}")

# Matrice de confusion pour la validation
cm_val = confusion_matrix(y_val_lab, y_val_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Reds',
            xticklabels=labels, yticklabels=labels)
plt.title('Matrice de confusion - Ensemble de VALIDATION')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Calcul des métriques globales sur la validation
precision_val = np.diag(cm_val) / np.sum(cm_val, axis=0)
recall_val = np.diag(cm_val) / np.sum(cm_val, axis=1)
f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

precision_val = np.nan_to_num(precision_val, nan=0.0)
recall_val = np.nan_to_num(recall_val, nan=0.0)
f1_score_val = np.nan_to_num(f1_score_val, nan=0.0)

print("\n=== MÉTRIQUES GLOBALES SUR VALIDATION ===")
print(f"Précision moyenne: {np.mean(precision_val):.4f}")
print(f"Rappel moyen: {np.mean(recall_val):.4f}")
print(f"F1-Score moyen: {np.mean(f1_score_val):.4f}")
print(f"Accuracy: {val_accuracy:.4f}")

# Teste
loaded_model = load_model("/content/drive/MyDrive/marine_sounds11_vgg19.h5")

# Convertir un nouveau fichier en spectrogramme
file_path = "/content/7501400O.wav"
img = wav_to_melspectrogram(file_path)
img = np.expand_dims(img, axis=0)

# Prédiction
prediction = loaded_model.predict(img)
pred_class = np.argmax(prediction)
print(f"Animal prédit : {labels[pred_class]} avec confiance {np.max(prediction):.2f}")