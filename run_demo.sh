#!/bin/bash
# Script pour générer des images démosaïquées de démonstration

echo "=== Génération d'images de démonstration MCAN ==="

# Activer l'environnement virtuel
source venv_mcan/bin/activate

# Démonstration avec image synthétique
echo "Démonstration avec image synthétique..."
python demo.py \
    --model "checkpoint1/mcan_model.pth" \
    --val_dir "CAVE_dataset/new_val" \
    --image "beads_ms" \
    --cuda

echo "Démonstration terminée !"