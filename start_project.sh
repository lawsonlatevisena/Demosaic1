#!/bin/bash
set -e

echo "🚀 Démarrage du projet MCTN"
echo "=============================="

PROJECT_DIR="/home/lawson/Documents/projet_Demosaic1"
cd "$PROJECT_DIR"

# Nettoyage du cache Python
echo "🧹 Nettoyage du cache Python..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Activation de l'environnement virtuel
echo "📦 Activation de l'environnement virtuel..."
if [ ! -d "venv_mcan" ]; then
    echo "❌ Environnement virtuel non trouvé. Création..."
    python3 -m venv venv_mcan
fi

source venv_mcan/bin/activate

# Mise à jour pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip --quiet 2>&1 | grep -v "already satisfied" || true

# Installation des packages essentiels
echo "📚 Installation des packages (cela peut prendre quelques minutes)..."
pip install --timeout=300 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
pip install --timeout=300 tqdm pandas numpy matplotlib scipy Pillow tifffile scikit-learn sewar 2>&1 | tail -5

# Vérification
echo ""
echo "✅ Vérification des installations..."
python3 << 'PYEOF'
import torch
import torchvision
import tqdm
import pandas
import numpy
import matplotlib
print("✓ PyTorch:", torch.__version__)
print("✓ Torchvision:", torchvision.__version__)
print("✓ CUDA disponible:", torch.cuda.is_available())
print("✓ Tous les packages sont installés!")
PYEOF

# Menu de démarrage
echo ""
echo "🎯 Options de démarrage:"
echo "1) Test rapide (2 epochs) - Recommandé"
echo "2) Test moyen (10 epochs)"
echo "3) Entraînement complet (85 epochs)"
echo "4) Reprendre depuis checkpoint"
echo ""
read -p "Choisissez une option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Démarrage du test rapide..."
        python3 main_MCAN.py \
            --batchSize 8 \
            --nEpochs 2 \
            --lr 2e-3 \
            --threads 0 \
            2>&1 | tee "logs/test_$(date +%Y%m%d_%H%M%S).log"
        ;;
    2)
        echo ""
        echo "🧪 Démarrage du test moyen..."
        python3 main_MCAN.py \
            --batchSize 8 \
            --nEpochs 10 \
            --lr 2e-3 \
            --threads 0 \
            2>&1 | tee "logs/test_$(date +%Y%m%d_%H%M%S).log"
        ;;
    3)
        echo ""
        echo "🏋️  Démarrage de l'entraînement complet..."
        python3 main_MCAN.py \
            --batchSize 16 \
            --nEpochs 85 \
            --lr 2e-3 \
            --threads 0 \
            2>&1 | tee "logs/training_$(date +%Y%m%d_%H%M%S).log"
        ;;
    4)
        echo ""
        echo "📂 Checkpoints disponibles:"
        ls -lht checkpoint1/*.pth | head -5
        echo ""
        read -p "Entrez le numéro d'epoch à reprendre: " epoch_num
        python3 main_MCAN.py \
            --batchSize 16 \
            --nEpochs 85 \
            --lr 2e-3 \
            --threads 0 \
            --resume "checkpoint1/De_happy_model_epoch_${epoch_num}.pth" \
            2>&1 | tee "logs/resume_$(date +%Y%m%d_%H%M%S).log"
        ;;
    *)
        echo "❌ Option invalide"
        exit 1
        ;;
esac

echo ""
echo "✅ Terminé!"
echo "📂 Résultats dans: checkpoint/"
