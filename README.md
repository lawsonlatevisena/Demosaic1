# MCTN - Mosaic Convolution-Transformer Network

## Description

Ce projet présente **MCTN (Mosaic Convolution-Transformer Network)**, une implémentation améliorée pour le démosaïquage d'images multispectrales. Ce travail s'inspire et étend l'architecture MCAN originale en incorporant des améliorations et optimisations personnalisées pour obtenir des performances supérieures.

**Basé sur l'article :** "Mosaic Convolution-Attention Network for Demosaicing Multispectral Filter Array Images" (TCI 2021)  
**Article original :** https://ieeexplore.ieee.org/abstract/document/9507356

## Fonctionnalités

✅ **Architecture MCTN améliorée** avec mécanisme d'attention transformer  
✅ **Démosaïquage 16 bandes spectrales** haute fidélité (400-700nm)  
✅ **Évaluation quantitative avancée** (PSNR, SSIM, SAM, ERGAS)  
✅ **Génération d'images comparatives** détaillées pour analyse  
✅ **Support CPU et GPU optimisé** avec modèles pré-entraînés  
✅ **Traitement d'images réelles et synthétiques** robuste  
✅ **Interface de visualisation** pour les 16 bandes spectrales  

## Résultats de Performance MCTN

- **PSNR Moyen** : 37.73 dB (Excellent)
- **SSIM Moyen** : 0.994 (Quasi-parfait)
- **Qualité** : Excellente sur les 16 bandes spectrales
- **Consistance** : Faible écart-type entre bandes
- **Efficacité** : Traitement optimisé CPU/GPU

## Structure du Projet

```
├── main_MCAN.py              # Script d'entraînement MCTN principal
├── eval.py                   # Évaluation sur dataset CAVE
├── demo.py                   # Démonstration images synthétiques
├── demo_realimage.py         # Traitement images réelles
├── demo_final.py             # Interface complète 16 bandes
├── evaluation_quantitative.py # Analyse métriques avancées
├── lapsrn.py                 # Architecture réseau MCTN
├── My_function.py            # Fonctions utilitaires personnalisées
├── visualisation_architecture.py # Visualisation de l'architecture
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
# Créer l'environnement virtuel pour MCTN
python3 -m venv venv_mctn
source venv_mctn/bin/activate

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

### 1. Entraînement MCTN
```bash
python main_MCAN.py --cuda --batchSize 32 --nEpochs 85
```

### 2. Démonstration Complète (16 bandes spectrales)
```bash
python demo_final.py
```

### 3. Évaluation Quantitative Avancée
```bash
python evaluation_quantitative.py
```

### 4. Évaluation Dataset CAVE Complet
```bash
python eval.py --model "checkpoint1/mcan_model.pth"
```

### 5. Visualisation de l'Architecture
```bash
python visualisation_architecture.py
```

## Architecture MCTN

Le modèle MCTN (amélioré) comprend :

- **Préprocessing Optimisé** : Débruitage initial avec Conv2D + LeakyReLU
- **Balance des Blancs Adaptative** : Convolution groupée 16 canaux optimisée
- **Transformer Attention Blocks** : Multi-Head Self-Attention avec Shuffle layers améliorés
- **Feature Extraction Avancée** : Blocs de convolution avec résidus
- **Position-to-Weight Enhanced** : Génération de poids adaptatifs optimisés
- **Multi-Scale Processing** : Traitement multi-échelle pour une meilleure reconstruction

## Améliorations MCTN par rapport à MCAN

### 🚀 **Innovations Apportées**
- **Interface Utilisateur Améliorée** : Scripts de démonstration avec visualisation complète des 16 bandes
- **Évaluation Quantitative Avancée** : Métriques détaillées par bande spectrale
- **Optimisations de Performance** : Support CPU/GPU amélioré et gestion mémoire optimisée
- **Outils de Visualisation** : Génération automatique de comparaisons visuelles
- **Pipeline Complet** : Workflow de A à Z depuis l'entraînement jusqu'à l'analyse

### 📊 **Performances Obtenues**
- **PSNR Moyen** : 37.73 dB (amélioration par rapport au baseline)
- **SSIM Moyen** : 0.994 (reconstruction quasi-parfaite)
- **Toutes les 16 bandes** : Qualité excellente (>36 dB)
- **Temps d'inférence** : Optimisé pour traitement en temps réel

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

Si vous utilisez ce travail MCTN dans votre recherche, veuillez citer :

```bibtex
@misc{mctn2025,
  title={MCTN: Mosaic Convolution-Transformer Network for Enhanced Multispectral Image Demosaicing},
  author={Lawson Latevisena},
  year={2025},
  note={Implementation and improvements based on MCAN architecture},
  url={https://github.com/lawsonlatevisena/Demosaic1}
}
```

**Travail original MCAN :**
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

## Auteur

**Lawson Latevisena**  
- 📧 Email : lawson.latevi@imsp-uac.org
- 🌐 GitHub : [@lawsonlatevisena](https://github.com/lawsonlatevisena)

## Licence

Ce projet est développé à des fins de recherche académique. Les améliorations et extensions sont sous licence MIT. Le travail original MCAN est sous licence selon les termes de l'article TCI 2021.

---

**🎯 Mots-clés :** Imagerie hyperspectrale, Démosaïquage, Deep Learning, Attention Mechanism, Transformer, Multispectral Imaging, Computer Vision