#!/bin/bash
# Script pour évaluer le modèle MCAN

echo "=== Évaluation du modèle MCAN ==="

# Activer l'environnement virtuel
source venv_mcan/bin/activate

# Évaluation avec le modèle le plus récent
echo "Évaluation en cours..."
python eval.py \
    --model "checkpoint1/mcan_model.pth" \
    --dataset "CAVE_dataset/new_val" \
    --cuda

echo "Évaluation terminée !"