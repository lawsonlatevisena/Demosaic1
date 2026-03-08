#!/bin/bash
# ========================================================================
# SCRIPT D'ENTRAÎNEMENT MCTN - AVEC SAUVEGARDE AUTOMATIQUE
# ========================================================================
# Ce script lance un nouvel entraînement avec :
# - Dossier de checkpoint UNIQUE (pas d'écrasement des anciens modèles)
# - Pas de chargement automatique de modèles existants
# - Sauvegarde avec timestamp pour identifier chaque entraînement
# ========================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 NOUVEL ENTRAÎNEMENT MCTN - DÉMOSAÏQUAGE SPECTRAL        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ========================================================================
# CONFIGURATION (Modifiez ces valeurs selon vos besoins)
# ========================================================================
EPOCHS=250              # Nombre d'epochs (250 recommandé pour bon équilibre)
BATCH_SIZE=8            # Taille du batch
LEARNING_RATE=2e-3      # Taux d'apprentissage
STEP=2000               # Décroissance LR tous les N epochs

echo "📋 Configuration de l'entraînement :"
echo "   ├─ Epochs          : $EPOCHS"
echo "   ├─ Batch Size      : $BATCH_SIZE"
echo "   ├─ Learning Rate   : $LEARNING_RATE"
echo "   └─ LR Decay Step   : $STEP"
echo ""

# Génération automatique du nom du dossier de checkpoint
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="checkpoint_train_${TIMESTAMP}_epochs${EPOCHS}"

echo "📁 Dossier de sauvegarde :"
echo "   └─ $CHECKPOINT_DIR/"
echo ""

# Créer le dossier logs si nécessaire
mkdir -p logs

echo "⚙️  Lancement de l'entraînement..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Lancement avec sauvegarde des logs
/usr/bin/python3 main_MCAN.py \
    --batchSize $BATCH_SIZE \
    --nEpochs $EPOCHS \
    --lr $LEARNING_RATE \
    --step $STEP \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    2>&1 | tee "logs/training_${TIMESTAMP}_${EPOCHS}epochs.log"

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Entraînement terminé avec succès !"
    echo ""
    echo "📂 Résultats sauvegardés dans :"
    echo "   ├─ Checkpoints : $CHECKPOINT_DIR/"
    echo "   ├─ Logs        : logs/training_${TIMESTAMP}_${EPOCHS}epochs.log"
    echo "   └─ CSV Stats   : $CHECKPOINT_DIR/1_train_results.csv"
    echo ""
    echo "💡 Pour évaluer ce modèle, utilisez :"
    echo "   python3 demo_final.py --checkpoint $CHECKPOINT_DIR/De_happy_model_epoch_$EPOCHS.pth"
else
    echo "❌ Erreur lors de l'entraînement (code: $EXIT_CODE)"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
