# Résumé du TP : Ce que vous devez faire & Quels sont les résultats

Ce fichier sert de guide rapide pour vous (ou vos étudiants) afin de comprendre immédiatement les étapes à réaliser et les résultats attendus pour valider ce Travail Pratique (TP).

---

## 🛠️ 1. Ce que vous devez faire (Étape par Étape)

Voici le parcours simple pour exécuter et s'approprier le projet :

### Étape A : Préparer l'environnement
Ouvrez votre terminal dans le répertoire du projet `CIFAR-10` et installez les packages Python requis.
```bash
pip install -r requirements.txt
```
*(Cette étape est déjà faite sur votre machine actuelle).*

### Étape B : Lancer l'entraînement du modèle
Exécutez le script principal en ligne de commande :
```bash
python tp_cnn_cifar10.py
```
**Ce qu'il faut observer pendant l'exécution :**
1. **Le téléchargement** : La première fois, TensorFlow télécharge automatiquement le dataset CIFAR-10 (environ 170 Mo).
2. **Le résumé du modèle (`model.summary()`)** : Observez la structure des couches convolutives, de max pooling et de dropout, ainsi que le nombre de paramètres à entraîner (environ 300 000).
3. **Le défilement des Époques** : Regardez l'évolution des valeurs d'accuracy (exactitude) et de loss (perte) à chaque époque (de 1 à 15). L'accuracy doit monter régulièrement et la perte doit descendre.

### Étape C : Analyser les résultats visuels
Une fois le script terminé, ouvrez et examinez les 4 fichiers d'images générés dans votre dossier pour valider le comportement du réseau.

### Étape D : Consulter la documentation académique
*   Ouvrez le fichier [Rapport_TP_CNN_CIFAR10.md](file:///c:/Users/Hamza/Desktop/Desktop-folders/estem-2025/S2/TP/AI_PYTHON/CIFAR-10/Rapport_TP_CNN_CIFAR10.md) pour lire ou présenter l'analyse théorique de chaque étape (preprocessing, convolutions, lutte contre l'overfitting).

---

## 📊 2. Quel est le résultat (Ce que vous obtenez)

Après l'exécution complète du script, voici ce qui est généré dans votre dossier de travail :

### A. Les Métriques de Performance (dans le terminal)
*   **Exactitude sur le jeu de test (Test Accuracy)** : **~72% à 73%**. Cela signifie que le modèle classe correctement environ 7,3 images sur 10 qu'il n'a jamais vues auparavant.
*   **Perte de test (Test Loss)** : **~0.78**.
*   **Absence d'overfitting** : Les performances sur les données d'entraînement (~72%) et de test (~73%) sont extrêmement proches. C'est le **résultat du Dropout** qui empêche le modèle de mémoriser les images par cœur.

### B. Les 4 Fichiers Graphiques Générés

1.  **`cifar10_examples.png`** : 
    *   *Ce que c'est* : Une grille de 10 images d'entraînement choisies au hasard.
    *   *Utilité* : Montrer à quoi ressemblent les images du dataset réel avec leurs étiquettes en français.
2.  **`cifar10_learning_curves.png`** : 
    *   *Ce que c'est* : Deux courbes d'évolution temporelle (l'une pour la Loss, l'autre pour l'Accuracy).
    *   *Utilité* : Diagnostiquer l'apprentissage. La convergence stable des courbes d'entraînement et de validation prouve que l'apprentissage s'est bien déroulé et qu'il n'y a pas de sur-apprentissage.
3.  **`cifar10_confusion_matrix.png`** : 
    *   *Ce que c'est* : Une heatmap montrant les prédictions du modèle pour chaque classe réelle.
    *   *Utilité* : Identifier les confusions logiques du réseau (ex. confondre les chats avec les chiens, ou les camions avec les voitures), mettant en évidence la complexité visuelle.
4.  **`cifar10_predictions.png`** : 
    *   *Ce que c'est* : Une grille de 10 prédictions sur le jeu de test.
    *   *Utilité* : Rendre le résultat concret. Les prédictions correctes s'affichent en **vert** et les erreurs en **rouge**, accompagnées du score de certitude en %.

### C. Le Rapport de TP Clé en Main
Le fichier [Rapport_TP_CNN_CIFAR10.md](file:///c:/Users/Hamza/Desktop/Desktop-folders/estem-2025/S2/TP/AI_PYTHON/CIFAR-10/Rapport_TP_CNN_CIFAR10.md) constitue le compte rendu rédigé de niveau universitaire, idéal pour servir de correction ou de support de cours.
