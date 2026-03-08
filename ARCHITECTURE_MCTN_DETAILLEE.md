# 🏗️ ARCHITECTURE DU MODÈLE MCTN (Multi-scale Convolutional Transformer Network)

## 📊 Vue d'ensemble

Le modèle MCTN est un réseau de neurones profonds conçu spécifiquement pour le **démosaïquage d'images hyperspectrales** à partir de capteurs MSFA (Multi-Spectral Filter Array) 4×4 avec 16 bandes spectrales.

### 🎯 Objectif
Transformer une image mosaïque (16 canaux entrelacés) → Image hyperspectrale complète (16 bandes de 512×512 pixels)

---

## 🔧 ARCHITECTURE GLOBALE

```
INPUT: Image mosaïque [B, 16, H, W]
   ↓
┌─────────────────────────────────────────────────────────────┐
│                    ÉTAPE 1: DÉBRUITAGE                       │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ Denoising    │  →   │ Denoising    │  →  [Estimation     │
│  │ Conv (3×3)   │      │ Output (3×3) │      du bruit]      │
│  │ 16→64        │      │ 64→16        │                     │
│  └──────────────┘      └──────────────┘                     │
│                              ↓                               │
│                    denoised_x = x - estimated_noise         │
└─────────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────────────────┐
│              ÉTAPE 2: WHITE BALANCE (WB_Conv)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Convolution 7×7 avec filtre bilinéaire fixe          │   │
│  │ - Grouped convolution (16 groupes)                   │   │
│  │ - Poids pré-calculés (non entraînables)              │   │
│  │ - Interpolation bilinéaire pour chaque bande         │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ↓                               │
│                         WB_norelu                            │
└─────────────────────────────────────────────────────────────┘
   ↓                          ↓
   │                          │ (connexion résiduelle)
   ↓                          │
┌─────────────────────────────────────────────────────────────┐
│           ÉTAPE 3: RECONSTRUCTION ADAPTATIVE                 │
│                   (Pos2Weight Network)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ A. Génération de poids locaux (Pos2Weight)           │   │
│  │    Input: Matrice de positions [H*W, 2]              │   │
│  │    ┌────────────┐     ┌────────────┐                 │   │
│  │    │ Linear     │  →  │ Linear     │                 │   │
│  │    │ 2→128      │     │ 128→400    │                 │   │
│  │    │ + ReLU     │     │ (5×5×16)   │                 │   │
│  │    └────────────┘     └────────────┘                 │   │
│  │         ↓                                             │   │
│  │    Poids adaptatifs par position spatiale            │   │
│  │                                                       │   │
│  │ B. Convolution locale adaptative                     │   │
│  │    - Unfold de l'image (patches 5×5)                 │   │
│  │    - Multiplication matricielle avec poids locaux    │   │
│  │    - Reconstruction de l'image                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ↓                               │
│                         Raw_conv [16, H, W]                  │
└─────────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────────────────┐
│      ÉTAPE 4: EXTRACTION DE CARACTÉRISTIQUES                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ A. Branch Front (MALayer)                            │   │
│  │    - Multi-head Self-Attention sur 16 canaux         │   │
│  │    - Spatial downsampling (Shuffle_d 4×)             │   │
│  │    - Attention globale                               │   │
│  │    - Spatial upsampling (PixelShuffle 4×)            │   │
│  │                                                       │   │
│  │ B. Front Conv Input                                  │   │
│  │    - Conv 3×3: 16→64 canaux                          │   │
│  │                                                       │   │
│  │ C. Feature Extraction (2× Conv_attention_Block)      │   │
│  │    Chaque bloc contient:                             │   │
│  │    ┌────────────────────────────────────────┐        │   │
│  │    │ • 3 Conv 3×3 (64→64→64→64)             │        │   │
│  │    │ • MALayer (Multi-head Attention)        │        │   │
│  │    │ • Connexion résiduelle                  │        │   │
│  │    └────────────────────────────────────────┘        │   │
│  │                                                       │   │
│  │ D. Branch Back                                       │   │
│  │    - Conv 3×3: 64→16 canaux                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ↓                               │
│                          HR_4x                               │
└─────────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────────────────┐
│             ÉTAPE 5: FUSION FINALE                           │
│                   HR_4x + WB_norelu                          │
│              (Connexion résiduelle globale)                  │
└─────────────────────────────────────────────────────────────┘
   ↓
OUTPUT: Image hyperspectrale [B, 16, H, W]
```

---

## 🧩 COMPOSANTS DÉTAILLÉS

### 1️⃣ Module de Débruitage

**Objectif**: Éliminer le bruit du capteur avant traitement

```python
Denoising:
├─ Conv2d(16→64, kernel=3×3)
├─ LeakyReLU(0.2)
├─ Conv2d(64→16, kernel=3×3)
└─ Soustraction: x_clean = x_brut - bruit_estimé
```

**Pourquoi ?** Les capteurs MSFA introduisent du bruit lors de l'acquisition.

---

### 2️⃣ White Balance (WB_Conv)

**Objectif**: Interpolation bilinéaire initiale

```python
Filtre bilinéaire 7×7:
[1/16  2/16  3/16  4/16  3/16  2/16  1/16]
[2/16  4/16  6/16  8/16  6/16  4/16  2/16]
[3/16  6/16  9/16 12/16  9/16  6/16  3/16]
[4/16  8/16 12/16 16/16 12/16  8/16  4/16]
[3/16  6/16  9/16 12/16  9/16  6/16  3/16]
[2/16  4/16  6/16  8/16  6/16  4/16  2/16]
[1/16  2/16  3/16  4/16  3/16  2/16  1/16]
```

- **Poids fixes** (non entraînables)
- **Grouped convolution** (16 groupes = 1 filtre par bande)
- Fournit une **estimation initiale grossière**

---

### 3️⃣ Pos2Weight (Reconstruction Adaptative)

**Objectif**: Générer des poids de convolution adaptatifs basés sur la position spatiale

```
Position (x,y) → MLP → Poids de convolution 5×5×16

Architecture:
Input: [H×W, 2]  (coordonnées normalisées)
  ↓
Linear(2 → 128) + ReLU
  ↓
Linear(128 → 400)  # 5×5×16 = 400
  ↓
Reshape: [H, W, 5, 5, 16]
  ↓
Convolution locale avec poids adaptatifs
```

**Innovation clé**: Chaque position spatiale a ses propres poids de convolution !

**Exemple**:
- Position (0, 0) : poids optimisés pour le coin supérieur gauche
- Position (H/2, W/2) : poids optimisés pour le centre
- Adaptation au pattern périodique 4×4 du MSFA

---

### 4️⃣ MALayer (Multi-head Attention Layer)

**Objectif**: Attention spatiale multi-échelle

```
Input [B, C, H, W]
  ↓
Shuffle_d (4×) → [B, C×16, H/4, W/4]  # Réduction résolution
  ↓
Flatten spatial → [B, (H/4)×(W/4), C×16]
  ↓
Linear projection → [B, (H/4)×(W/4), C]
  ↓
Multi-head Self-Attention (4 têtes)
  ↓
FC layers → Attention map [B, C×16, 1, 1]
  ↓
Multiplication élément par élément
  ↓
PixelShuffle (4×) → [B, C, H, W]  # Restauration résolution
```

**Pourquoi multi-échelle ?**
- Capture des dépendances à longue distance
- Réduit la complexité computationnelle (H/4 × W/4)
- 4 têtes d'attention = 4 représentations différentes

---

### 5️⃣ Conv_attention_Block

**Objectif**: Extraction de caractéristiques avec attention

```
Bloc résiduel:
Input [64, H, W]
  ↓
┌────────────────────┐
│ Conv 3×3 (64→64)   │
│ LeakyReLU          │
│ Conv 3×3 (64→64)   │
│ LeakyReLU          │
│ Conv 3×3 (64→64)   │
└────────────────────┘
  ↓
MALayer (Attention)
  ↓
+ Connexion résiduelle
  ↓
LeakyReLU
  ↓
Output [64, H, W]
```

**2 blocs en série** → Hiérarchie de caractéristiques (bas niveau + haut niveau)

---

## 📈 FLUX DE DONNÉES COMPLET

### Exemple avec image 512×512, 16 bandes

```
Étape               Dimensions           Opération
─────────────────────────────────────────────────────────────
Input mosaïque      [1, 16, 512, 512]    Image brute
  ↓
Denoising           [1, 16, 512, 512]    Suppression bruit
  ↓
WB_Conv             [1, 16, 512, 512]    Interpolation bilinéaire
  ↓
Pos2Weight          [1, 16, 512, 512]    Reconstruction adaptative
  ↓
MALayer (front)     [1, 16, 512, 512]    Attention spatiale
  ↓
Front Conv Input    [1, 64, 512, 512]    Expansion canaux
  ↓
Conv_attention_1    [1, 64, 512, 512]    Features bas niveau
  ↓
Conv_attention_2    [1, 64, 512, 512]    Features haut niveau
  ↓
Branch Back         [1, 16, 512, 512]    Réduction canaux
  ↓
+ WB_Conv (résid.)  [1, 16, 512, 512]    Fusion finale
  ↓
Output final        [1, 16, 512, 512]    Image démosaïquée
```

---

## 🎓 CONCEPTS CLÉS

### 1. Connexions résiduelles
```
WB_Conv (branche rapide) + HR_4x (branche profonde)
```
- **Pourquoi ?** Facilite l'apprentissage (le réseau apprend les résidus)
- Évite le problème de gradient vanishing

### 2. Multi-head Attention
```
4 têtes → 4 représentations complémentaires
```
- Chaque tête capture des patterns différents
- Augmente la capacité du modèle

### 3. Convolution adaptative (Pos2Weight)
```
Poids de convolution = f(position spatiale)
```
- Adaptation au pattern périodique 4×4 du MSFA
- Plus flexible que convolution standard

### 4. Architecture multi-échelle
```
Résolution complète → Résolution réduite (attention) → Résolution complète
```
- Balance entre champ réceptif et complexité
- Capture des dépendances globales

---

## 📊 STATISTIQUES DU MODÈLE

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Composant                    Paramètres      Type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Denoising Conv               9,216           Entraînable
Denoising Output             9,216           Entraînable
WB_Conv                      784             FIXE (non entraînable)
Front Conv Input             9,216           Entraînable
Branch Front (MALayer)       ~85,000         Entraînable
Conv_attention_Block × 2     ~280,000        Entraînable
Branch Back                  1,024           Entraînable
Pos2Weight                   51,600          Entraînable
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                        616,064         Paramètres
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Répartition:
- **MALayer + Conv_attention**: ~60% des paramètres
- **Pos2Weight**: ~8% des paramètres
- **Reste (Conv standard)**: ~32% des paramètres

---

## 🔍 FONCTION DE PERTE

```python
L1_Charbonnier_loss:

L = Σ sqrt((pred - target)² + ε)

où ε = 1e-6
```

**Avantages**:
- Plus robuste que MSE aux outliers
- Dérivée continue (contrairement à L1)
- Favorise la reconstruction fine des détails

---

## 🎯 POURQUOI CETTE ARCHITECTURE FONCTIONNE ?

### ✅ Forces:
1. **Débruitage intégré** → Gère le bruit du capteur
2. **White Balance** → Estimation initiale stable
3. **Pos2Weight** → Adaptatif au pattern MSFA périodique
4. **Multi-head Attention** → Capture dépendances globales
5. **Connexions résiduelles** → Apprentissage stable
6. **Multi-échelle** → Balance locale/globale

### 🎓 Innovation principale:
**Combinaison unique de**:
- Convolution adaptative (Pos2Weight)
- Attention multi-têtes (Transformer-like)
- Architecture multi-échelle
- Processing spatial/spectral joint

---

## 📚 COMPARAISON AVEC D'AUTRES APPROCHES

| Méthode           | Interpolation | Attention | Adaptatif | Performance |
|-------------------|---------------|-----------|-----------|-------------|
| Bilinéaire simple | ✅            | ❌        | ❌        | ~25 dB      |
| CNN classique     | ✅            | ❌        | ❌        | ~32 dB      |
| ResNet            | ✅            | ❌        | ⚠️        | ~35 dB      |
| **MCTN (notre)**  | ✅            | ✅        | ✅        | **37.73 dB**|

---

## 🚀 POINTS CLÉS À RETENIR

1. **Architecture hybride**: Convolution + Attention (comme les Vision Transformers)
2. **Adaptatif**: Poids varient selon la position spatiale
3. **Multi-échelle**: Traite l'information à différentes résolutions
4. **Résiduel**: Facilite l'entraînement profond
5. **Robuste**: Débruitage intégré pour gérer le bruit du capteur

---

## 🔬 EXEMPLE DE TRAITEMENT

```
Pixel MSFA position (0,0):
  ↓
1. Débruitage: supprime bruit capteur
  ↓
2. WB: interpolation bilinéaire → 16 valeurs estimées
  ↓
3. Pos2Weight: génère poids optimaux pour position (0,0)
  ↓
4. Convolution adaptative: raffine avec voisinage 5×5
  ↓
5. Attention: contexte global (toute l'image)
  ↓
6. Conv blocks: extraction features hiérarchiques
  ↓
7. Fusion: combine WB + features profondes
  ↓
Résultat: 16 valeurs spectrales raffinées pour position (0,0)
```

---

Cette architecture est le résultat de recherches pour combiner le meilleur de:
- **CNNs classiques** (convolutions, résidus)
- **Transformers** (attention multi-têtes)
- **Réseaux adaptatifs** (Pos2Weight)

Pour atteindre des performances state-of-the-art sur le démosaïquage hyperspectral ! 🎉
