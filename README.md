# MCAN - Mosaic Convolution-Attention Network

## Description

Ce projet implémente le réseau **MCAN (Mosaic Convolution-Attention Network)** pour le démosaïquage d'images multispectrales. Il s'agit de l'implémentation officielle PyTorch de l'article "Mosaic Convolution-Attention Network for Demosaicing Multispectral Filter Array Images" (TCI 2021).

**Article original :** https://ieeexplore.ieee.org/abstract/document/9507356

## Fonctionnalités

✅ **Entraînement du modèle MCAN** avec attention multi-head  
✅ **Démosaïquage 16 bandes spectrales** (400-700nm)  
✅ **Évaluation quantitative** (PSNR, SSIM, SAM, ERGAS)  
✅ **Génération d'images comparatives** pour analyse visuelle  
✅ **Support CPU et GPU** avec modèles pré-entraînés  
✅ **Traitement d'images réelles** et synthétiques  

## Résultats de Performance

- **PSNR Moyen** : 37.73 dB
- **SSIM Moyen** : 0.994
- **Qualité** : Excellente sur les 16 bandes spectrales

## Structure du Projet

```
├── main_MCAN.py              # Script d'entraînement principal
├── eval.py                   # Évaluation sur dataset CAVE
├── demo.py                   # Démonstration images synthétiques
├── demo_realimage.py         # Traitement images réelles
├── demo_final.py             # Script complet avec 16 bandes
├── evaluation_quantitative.py # Analyse des métriques
├── lapsrn.py                 # Architecture du réseau MCAN
├── My_function.py            # Fonctions utilitaires
├── requirements.txt          # Dépendances Python
└── README.md                 # Cette documentation
```

## Installation

### Prérequis système
```bash
sudo apt-get install libtiff4-dev
```

### Environnement Python
```bash
# Créer l'environnement virtuel
python3 -m venv venv_mcan
source venv_mcan/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- matplotlib >= 3.3.0
- tifffile >= 2021.7.2
- scikit-learn
- sewar

## Utilisation

### 1. Entraînement
```bash
python main_MCAN.py --cuda --batchSize 32 --nEpochs 85
```

### 2. Démonstration Complète (16 bandes)
```bash
python demo_final.py
```

### 3. Évaluation Quantitative
```bash
python evaluation_quantitative.py
```

### 4. Évaluation Dataset Complet
```bash
python eval.py --model "checkpoint1/mcan_model.pth"
```

## Architecture MCAN

Le modèle MCAN comprend :

- **Débruitage Initial** : Conv2D + LeakyReLU
- **Balance des Blancs** : Convolution groupée 16 canaux
- **Blocs d'Attention** : Multi-Head Self-Attention avec Shuffle layers
- **Blocs de Convolution** : Extraction de caractéristiques
- **Position-to-Weight** : Génération de poids adaptatifs

## Résultats

### Images Générées
- `resultats_demo/comparaison_bande_1.png` à `comparaison_bande_16.png`
- `resultats_demo/image_demosaiquee_16_bandes.tif`
- `resultats_demo/comparaison_rgb_composite.png`

### Métriques
- Fichier CSV : `resultats_demo/metriques_16_bandes.csv`
- PSNR > 36 dB sur toutes les bandes
- SSIM > 0.99 (quasi-parfait)

## Citation

Si vous trouvez ce code et ces datasets utiles dans votre recherche, veuillez citer :

```bibtex
@article{feng2021mosaic,
  title={Mosaic Convolution-Attention Network for Demosaicing Multispectral Filter Array Images},
  author={Feng, Kai and Zhao, Yongqiang and Chan, Jonathan Cheung-Wai and Kong, Seong G and Zhang, Xun and Wang, Binglu},
  journal={IEEE Transactions on Computational Imaging},
  volume={7},
  pages={864--878},
  year={2021},
  publisher={IEEE}
}
```

## Licence

Ce projet est sous licence selon les termes de l'article original TCI 2021.
