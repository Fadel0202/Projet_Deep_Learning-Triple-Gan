# Projet Triple-GAN pour la Classification Semi-supervisée sur MNIST

## Auteurs
- Mouhamed SAMB
- Moustapha KEBE
- Mouhamadou lamine GNING
- Birahim TEWE

Master 2 MLSD/AMSD - Centre Borelli, Université Paris Cité (2024/2025)

## Description du projet

Ce projet implémente le modèle Triple-GAN pour la classification semi-supervisée d'images MNIST en utilisant seulement 100 exemples étiquetés. Le Triple-GAN est une extension des réseaux antagonistes génératifs (GANs) classiques qui introduit un troisième acteur dans l'architecture pour améliorer la génération conditionnelle et la classification.

## Architecture du Triple-GAN

Le Triple-GAN se compose de trois réseaux distincts:

1. **Générateur (G)**: Apprend la distribution conditionnelle p_g(x|y) pour générer des images à partir d'étiquettes.
2. **Classificateur (C)**: Approxime p_c(y|x) pour prédire des étiquettes à partir d'images.
3. **Discriminateur (D)**: Identifie si une paire (x, y) provient de la distribution réelle ou est générée.

## Résultats

- **Précision de classification**: 85.37% (±0.58%) avec seulement 100 exemples étiquetés
- **Comparaison**: Cette performance surpasse le CNN baseline (60.5%) et même les résultats de l'article original (83.01%)
- **Qualité des images générées**: Le générateur produit des chiffres clairement identifiables avec contrôle des classes

## Contenu du Notebook

Le notebook est organisé comme suit:

1. **Installation des dépendances**
2. **Importation des bibliothèques**
3. **Configuration des hyperparamètres**
4. **Préparation des données**
   - Chargement du dataset MNIST
   - Séparation en exemples étiquetés (100) et non-étiquetés
   - Création des dataloaders
5. **Implémentation des modèles**
   - Générateur
   - Discriminateur
   - Classificateur
6. **Définition des fonctions d'entraînement**
   - Boucle d'entraînement principale
   - Fonction d'évaluation
   - Fonctions de visualisation
7. **Entraînement du modèle**
8. **Évaluation et visualisation**
   - Courbes d'apprentissage
   - Génération d'images par classe
   - Matrice de confusion

## Comment exécuter le projet

### Prérequis
- Python 3.7+
- PyTorch 1.7+
- Torchvision
- Matplotlib
- NumPy
- tqdm

### Installation

```bash
pip install torch torchvision matplotlib numpy tqdm
```

### Exécution

1. Clonez ce dépôt:
```bash
git clone https://github.com/[votre-username]/triple-gan-mnist.git
cd triple-gan-mnist
```

2. Ouvrez le notebook dans Jupyter:
```bash
jupyter notebook Triple_GAN_MNIST.ipynb
```

3. Exécutez toutes les cellules du notebook séquentiellement.

## Hyperparamètres

Les hyperparamètres utilisés pour l'entraînement sont:

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64 |
| Dimension du bruit (Z_DIM) | 100 |
| Dimension cachée (HIDDEN_DIM) | 256 |
| Learning rate | 0.0002 |
| Beta1 (Adam) | 0.5 |
| Nombre d'époques | 200 |
| α | 0.3 |

## Structure du code

### Générateur
```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, HIDDEN_DIM)
        
        self.model = nn.Sequential(
            nn.Linear(Z_DIM + HIDDEN_DIM, HIDDEN_DIM * 4),
            nn.BatchNorm1d(HIDDEN_DIM * 4),
            nn.ReLU(True),
            
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM * 8),
            nn.BatchNorm1d(HIDDEN_DIM * 8),
            nn.ReLU(True),
            
            nn.Linear(HIDDEN_DIM * 8, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        z = torch.cat([z, label_embedding], dim=1)
        img = self.model(z)
        return img.view(-1, 1, 28, 28)
```

### Discriminateur
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, HIDDEN_DIM)
        
        self.model = nn.Sequential(
            nn.Linear(784 + HIDDEN_DIM, HIDDEN_DIM * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(-1, 784)
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)
```

### Classificateur
```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_DIM, 10)
        )
    
    def forward(self, x):
        return self.model(x)
```

## Références

1. Li, C., Xu, K., Zhu, J., & Zhang, B. (2017). Triple generative adversarial nets. In Advances in neural information processing systems
2. Goodfellow, I., et al. (2014). Generative adversarial nets. In Advances in neural information processing systems
3. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE
4. Salimans, T., et al. (2016). Improved techniques for training GANs. In Advances in neural information processing systems

## Licence

Ce projet est disponible sous licence MIT.
