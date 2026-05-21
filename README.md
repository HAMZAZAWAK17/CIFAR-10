# TP Intelligence Artificielle : Classification d'Images CIFAR-10 avec un CNN

Ce dépôt contient le code source et la documentation complète pour réaliser un Travail Pratique (TP) universitaire sur la classification d'images à l'aide d'un Réseau de Neurones Convolutif (CNN) en utilisant le dataset **CIFAR-10** sous Python avec **TensorFlow / Keras**.

## 📁 Contenu du Projet

1. **`tp_cnn_cifar10.py`** : Le code Python complet, propre, structuré et commenté de manière pédagogique. Il gère le chargement du dataset, le prétraitement, l'architecture du réseau, l'entraînement, l'évaluation et la génération des graphiques.
2. **`Rapport_TP_CNN_CIFAR10.md`** : Le rapport académique rédigé en français (style TP universitaire). Il détaille la théorie des CNNs, justifie les étapes de prétraitement, explique le rôle de chaque couche de l'architecture, et analyse en détail les performances.
3. **`requirements.txt`** : Fichier recensant les dépendances Python nécessaires au bon fonctionnement du TP.

---

## 🛠️ Installation et Configuration

Pour exécuter ce projet, vous devez disposer de Python (version 3.8 ou supérieure de préférence).

### 1. Cloner ou Ouvrir le Projet
Placez-vous dans le répertoire du projet :
```bash
cd CIFAR-10
```

### 2. Installer les Dépendances
Nous vous recommandons de créer un environnement virtuel (optionnel mais conseillé) avant d'installer les bibliothèques nécessaires à l'aide du fichier `requirements.txt` :

```bash
# Installation des packages requis
pip install -r requirements.txt
```

---

## 🚀 Exécution du Projet

Pour lancer l'ensemble du pipeline (chargement des données, entraînement du CNN, évaluation et génération des graphiques), exécutez simplement la commande suivante dans votre terminal :

```bash
python tp_cnn_cifar10.py
```

Le script commencera par charger automatiquement le dataset CIFAR-10 (il le téléchargera lors de la première exécution). Il affichera ensuite le résumé textuel du modèle et lancera l'entraînement sur 15 époques.

---

## 📊 Fichiers Graphiques Générés

À la fin de l'exécution, le script génère automatiquement 4 images clés dans le répertoire courant :

1. **`cifar10_examples.png`** : Une grille de 10 images tirées aléatoirement du dataset d'entraînement pour visualiser les images d'entrée avec leurs noms de classe.
2. **`cifar10_learning_curves.png`** : Les courbes comparatives d'Exactitude (Accuracy) et de Perte (Loss) entre le jeu d'entraînement et de validation, idéales pour diagnostiquer le sur-apprentissage (overfitting).
3. **`cifar10_confusion_matrix.png`** : Une superbe Heatmap illustrant la matrice de confusion. Elle montre quelles classes d'images sont les plus souvent confondues par le modèle (ex : chats et chiens).
4. **`cifar10_predictions.png`** : Une grille d'exemples pratiques de prédictions sur le jeu de test. Les titres sont affichés en **vert** si la prédiction est correcte et en **rouge** en cas d'erreur.

---

## 🎓 Objectifs Pédagogiques du TP

En complétant ce TP, les étudiants apprendront à :
- Manipuler un dataset d'images réelles avec NumPy et TensorFlow.
- Normaliser des pixels d'image pour stabiliser l'apprentissage.
- Comprendre les principes des couches de Convolution (`Conv2D`), de Pooling (`MaxPooling2D`), d'Aplatissement (`Flatten`), de Classification (`Dense`) et de Sortie (`Softmax`).
- Mettre en place des méthodes simples de lutte contre l'overfitting comme le **Dropout**.
- Analyser rigoureusement les résultats d'un modèle de classification grâce aux courbes de convergence et à la **matrice de confusion**.
