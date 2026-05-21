# -*- coding: utf-8 -*-
"""
TP d'Intelligence Artificielle : Classification d'Images CIFAR-10 avec un CNN
=============================================================================
Ce script est conçu comme un support pédagogique pour les étudiants débutants
en Deep Learning. Il détaille chaque étape du pipeline de machine learning :
chargement, exploration, prétraitement, architecture, entraînement, évaluation
et analyse des performances d'un Réseau de Neurones Convolutif (CNN).

Auteur: Antigravity (Assistant IA Google DeepMind)
Date: Mai 2026
"""

# =====================================================================
# 1. IMPORTATION DES BIBLIOTHÈQUES
# =====================================================================
# Nous importons les outils nécessaires :
# - tensorflow/keras pour la création et l'entraînement du réseau de neurones.
# - numpy pour la manipulation des tableaux de données.
# - matplotlib & seaborn pour la visualisation des résultats et des graphiques.
# - scikit-learn pour le calcul de la matrice de confusion.
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Désactivation des logs TensorFlow verbeux pour plus de clarté
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TensorFlow Version:", tf.__version__)
print("Vérification GPU disponible:", tf.config.list_physical_devices('GPU'))

# =====================================================================
# 2. CHARGEMENT ET PRÉSENTATION DES DONNÉES
# =====================================================================
# CIFAR-10 est un dataset de référence contenant 60 000 images de 32x32 pixels
# en couleur (3 canaux de couleur : Rouge, Vert, Bleu - RGB).
# Il contient 10 classes d'objets ou d'animaux.
print("\n--- Chargement automatique du dataset CIFAR-10 ---")
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Suppression des dimensions inutiles des labels (passage de (N, 1) à (N,))
y_train = y_train.squeeze()
y_test = y_test.squeeze()

print(f"Données d'entraînement : X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Données de test         : X_test = {X_test.shape}, y_test = {y_test.shape}")

# Définition des classes textuelles en français pour la lisibilité
class_names = [
    'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf',
    'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
]

# =====================================================================
# 3. VISUALISATION D'IMAGES EXEMPLES
# =====================================================================
# Avant de prétraiter, il est important d'inspecter visuellement les données.
# Nous allons afficher et sauvegarder une grille d'exemples aléatoires du dataset.
print("\n--- Génération et sauvegarde d'images exemples (cifar10_examples.png) ---")
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    # Sélectionner une image aléatoire
    idx = random.randint(0, len(X_train) - 1)
    plt.imshow(X_train[idx])
    plt.title(class_names[y_train[idx]], fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.savefig('cifar10_examples.png')
plt.close()
print("Images exemples sauvegardées avec succès !")

# =====================================================================
# 4. PRÉTRAITEMENT DES DONNÉES (PREPROCESSING)
# =====================================================================
# Les pixels d'une image couleur ont des valeurs entières comprises entre 0 et 255.
# Pour faciliter l'apprentissage et la convergence du réseau, nous devons normaliser
# ces valeurs pour les ramener dans l'intervalle [0, 1] en divisant par 255.0.
# Ce processus de normalisation (ou mise à l'échelle) évite que les gradients n'explosent.
print("\n--- Prétraitement et normalisation des données ---")
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

print("Valeurs minimales et maximales après normalisation :")
print(f"Entraînement : Min = {X_train_norm.min()}, Max = {X_train_norm.max()}")
print(f"Test         : Min = {X_test_norm.min()}, Max = {X_test_norm.max()}")

# =====================================================================
# 5. CONSTRUCTION DE L'ARCHITECTURE CNN SIMPLE ET PÉDAGOGIQUE
# =====================================================================
# Nous allons concevoir un modèle séquentiel (les couches s'empilent l'une après l'autre).
# Ce CNN est composé de deux blocs d'extraction de caractéristiques (Convolution + Pooling),
# suivis d'un bloc de classification (Flatten + Dense) avec du Dropout pour le contrôle de l'overfitting.
print("\n--- Création du modèle CNN ---")

model = models.Sequential([
    # --- BLOC 1 : Extraction de caractéristiques de bas niveau (formes simples, contours)
    # Couche de Convolution 2D : applique 32 filtres de taille 3x3 sur l'image d'entrée.
    # L'activation 'relu' (Rectified Linear Unit) introduit de la non-linéarité.
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), name='Conv1'),
    
    # Couche de Max Pooling 2D : réduit les dimensions spatiales de moitié (de 30x30 à 15x15)
    # en conservant la valeur maximale sur des fenêtres de 2x2. Réduit le coût de calcul.
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool1'),
    
    # --- BLOC 2 : Extraction de caractéristiques de haut niveau (motifs complexes)
    # Couche de Convolution 2D plus profonde : 64 filtres de taille 3x3.
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='Conv2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool2'),
    
    # --- RÉGULARISATION : Dropout
    # Le Dropout désactive aléatoirement un pourcentage de neurones à chaque étape de
    # l'entraînement (ici 25%), ce qui force le réseau à apprendre des représentations
    # redondantes et robustes, évitant ainsi le sur-apprentissage (overfitting).
    layers.Dropout(0.25, name='Dropout_Conv'),
    
    # --- BLOC 3 : Classification (Réseau de neurones classique ou Fully Connected)
    # Couche de Flatten (Aplatissement) : convertit la matrice 2D résultante des convolutions
    # en un vecteur à 1 dimension pour pouvoir alimenter la couche Dense suivante.
    layers.Flatten(name='Aplatissement'),
    
    # Couche Dense (Entièrement connectée) intermédiaire de 128 neurones avec activation ReLU.
    layers.Dense(units=128, activation='relu', name='Dense_Dense'),
    
    # Dropout plus fort (50%) avant la couche finale pour protéger la partie classification.
    layers.Dropout(0.5, name='Dropout_Dense'),
    
    # Couche finale de Sortie : 10 neurones (1 pour chaque classe).
    # L'activation 'softmax' transforme les scores bruts en probabilités. La somme des 10 sorties vaut 1.
    layers.Dense(units=10, activation='softmax', name='Sortie_Softmax')
])

# Affichage du résumé de l'architecture
# Ce tableau est essentiel pour comprendre les dimensions et le nombre de paramètres à chaque couche.
model.summary()

# =====================================================================
# 6. COMPILATION DU MODÈLE
# =====================================================================
# La compilation configure le processus d'apprentissage du réseau :
# - L'optimiseur (ici 'adam') ajuste les poids du réseau pour minimiser l'erreur.
# - La fonction de perte (loss) mesure l'écart entre la prédiction et la réalité. 
#   On utilise 'sparse_categorical_crossentropy' car nos classes sont encodées par des entiers [0-9].
# - La métrique suivie est l'exactitude ('accuracy') : pourcentage d'images bien classifiées.
print("\n--- Compilation du modèle ---")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Modèle compilé avec succès !")

# =====================================================================
# 7. ENTRAÎNEMENT DU MODÈLE
# =====================================================================
# Nous entraînons le réseau sur 15 époques.
# validation_data permet de suivre en temps réel la capacité de généralisation du modèle
# sur des données qu'il ne voit pas durant les calculs de gradient.
EPOCHS = 15
BATCH_SIZE = 64

print(f"\n--- Début de l'entraînement ({EPOCHS} époques, Batch Size = {BATCH_SIZE}) ---")
history = model.fit(
    X_train_norm, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_norm, y_test),
    verbose=1
)

# =====================================================================
# 8. ÉVALUATION SUR LE DATASET DE TEST
# =====================================================================
print("\n--- Évaluation du modèle sur le dataset de test ---")
test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"Performance sur le jeu de test :")
print(f"  - Perte (Loss)      = {test_loss:.4f}")
print(f"  - Exactitude (Acc)  = {test_acc * 100:.2f}%")

# =====================================================================
# 9. GÉNÉRATION DES GRAPHIQUES DE PERFORMANCE
# =====================================================================
# Les courbes de loss et d'accuracy permettent d'évaluer la convergence
# et de diagnostiquer visuellement l'overfitting.
print("\n--- Génération et sauvegarde des courbes d'apprentissage (cifar10_learning_curves.png) ---")
plt.figure(figsize=(12, 5))

# Graphique d'Exactitude (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entraînement', color='#3498db', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', color='#e74c3c', linewidth=2)
plt.title("Évolution de l'Exactitude (Accuracy)")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# Graphique de Perte (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entraînement', color='#3498db', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', color='#e74c3c', linewidth=2)
plt.title("Évolution de la Perte (Loss)")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('cifar10_learning_curves.png')
plt.close()
print("Courbes d'apprentissage sauvegardées !")

# =====================================================================
# 10. GÉNÉRATION DE LA MATRICE DE CONFUSION
# =====================================================================
# La matrice de confusion montre la distribution des prédictions pour chaque classe réelle.
# Elle permet d'identifier précisément les confusions les plus fréquentes (ex: Chat pris pour un Chien).
print("\n--- Calcul et sauvegarde de la matrice de confusion (cifar10_confusion_matrix.png) ---")
# Obtenir les prédictions du modèle
y_pred_probs = model.predict(X_test_norm, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculer la matrice
cm = confusion_matrix(y_test, y_pred)

# Affichage sous forme de Heatmap élégante avec Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title("Matrice de Confusion sur le jeu de test", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Classes Prédites", fontsize=12, labelpad=10)
plt.ylabel("Classes Réelles", fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('cifar10_confusion_matrix.png')
plt.close()
print("Matrice de confusion sauvegardée !")

# Affichage du rapport textuel détaillé des performances (Precision, Recall, F1-Score)
print("\n--- Rapport de Classification Textuel ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# =====================================================================
# 11. VISUALISATION DES PRÉDICTIONS DE TEST
# =====================================================================
# Pour illustrer concrètement le fonctionnement du réseau, nous allons sélectionner
# 10 images du jeu de test et afficher le résultat de la prédiction du modèle.
# La légende s'affiche en vert si la prédiction est correcte, en rouge sinon.
print("\n--- Génération et sauvegarde d'exemples de prédictions (cifar10_predictions.png) ---")
plt.figure(figsize=(12, 6))
indices_random = random.sample(range(len(X_test_norm)), 10)

for i, idx in enumerate(indices_random):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx])
    
    pred_label = y_pred[idx]
    real_label = y_test[idx]
    
    # Couleur : vert si correct, rouge si incorrect
    color = '#2ecc71' if pred_label == real_label else '#e74c3c'
    
    # Nom prédit et pourcentage de certitude (probabilité)
    pred_prob = y_pred_probs[idx][pred_label] * 100
    
    title_text = f"Prédit: {class_names[pred_label]}\nRéel: {class_names[real_label]}\n({pred_prob:.1f}%)"
    plt.title(title_text, color=color, fontsize=10, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('cifar10_predictions.png')
plt.close()
print("Graphique des prédictions de test sauvegardé !")

print("\n=====================================================================")
print("FÉLICITATIONS ! Le script du TP s'est exécuté en entier.")
print("Les fichiers d'images suivants ont été générés dans votre répertoire :")
print("1. cifar10_examples.png        -> Exemples aléatoires du dataset de départ")
print("2. cifar10_learning_curves.png  -> Évolution de la Loss et de l'Accuracy")
print("3. cifar10_confusion_matrix.png -> Matrice de confusion détaillée")
print("4. cifar10_predictions.png     -> Prédictions visuelles avec couleur de statut")
print("=====================================================================")
