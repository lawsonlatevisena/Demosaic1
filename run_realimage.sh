#!/bin/bash
# Script pour traiter des images réelles multispectrales

echo "=== Traitement d'images réelles multispectrales ==="

# Activer l'environnement virtuel
source venv_mcan/bin/activate

# Traitement d'images réelles
echo "Traitement d'images réelles en cours..."
python demo_realimage.py \
    --model "checkpoint1/mcan_model.pth" \
    --msfa_size 4 \
    --cuda

echo "Traitement terminé ! Vérifiez les résultats."