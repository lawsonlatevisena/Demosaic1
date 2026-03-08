#!/bin/bash
# Script d'entraînement de 300 epochs depuis zéro
# Sans continuation - Entraînement complet nouveau

echo "🚀 Démarrage de l'entraînement 300 epochs depuis zéro"
echo "=================================================="
echo "📋 Configuration:"
echo "   - Epochs: 300"
echo "   - Batch Size: 8"
echo "   - Learning Rate: 2e-3"
echo "   - Checkpoint: Nouveau (pas de continuation)"
echo "   - Dossier de sauvegarde: checkpoint/"
echo "=================================================="
echo ""

# Sauvegarde de l'ancien checkpoint si existe
if [ -f "checkpoint/1_train_results.csv" ]; then
    echo "💾 Sauvegarde de l'ancien historique d'entraînement..."
    mv checkpoint/1_train_results.csv checkpoint/1_train_results_backup_$(date +%Y%m%d_%H%M%S).csv
fi

if [ -f "checkpoint/1_opt.csv" ]; then
    mv checkpoint/1_opt.csv checkpoint/1_opt_backup_$(date +%Y%m%d_%H%M%S).csv
fi

# Lancement de l'entraînement
echo "🎯 Lancement de l'entraînement..."
/usr/bin/python3 main_MCAN.py \
    --batchSize 8 \
    --nEpochs 300 \
    --lr 2e-3 \
    --step 2000 \
    2>&1 | tee logs/training_300epochs_from_scratch_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Entraînement terminé !"
echo "📁 Checkpoint sauvegardé dans: checkpoint/De_happy_model_epoch_300.pth"
echo "📊 Historique dans: checkpoint/1_train_results.csv"
