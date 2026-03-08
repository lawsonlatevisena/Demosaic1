# 📚 GUIDE D'UTILISATION - ENTRAÎNEMENT MCTN

## ✅ Modifications apportées

Le script `main_MCAN.py` a été modifié pour **éviter l'écrasement des modèles** :

### 🔧 Changements principaux

1. **Dossier unique par entraînement** : Chaque entraînement crée un nouveau dossier avec timestamp
   - Format : `checkpoint_train_YYYYMMDD_HHMMSS_epochsN/`
   - Exemple : `checkpoint_train_20251130_160530_epochs250/`

2. **Pas de chargement automatique** : Le paramètre `--resume=""` est vide par défaut
   - Les anciens modèles ne sont JAMAIS chargés automatiquement
   - Pour continuer un entraînement, il faut explicitement spécifier le checkpoint

3. **Sauvegarde organisée** :
   ```
   checkpoint_train_20251130_160530_epochs250/
   ├── De_happy_model_epoch_250.pth    # Checkpoint
   ├── 1_train_results.csv              # Historique d'entraînement
   └── 1_opt.csv                        # Configuration
   ```

## 🚀 Comment lancer un nouvel entraînement

### Méthode 1 : Script automatique (RECOMMANDÉ)

```bash
chmod +x train_new.sh
./train_new.sh
```

Ce script :
- ✅ Crée automatiquement un dossier unique
- ✅ Sauvegarde les logs
- ✅ Ne touche PAS aux anciens modèles
- ✅ Affiche un résumé à la fin

### Méthode 2 : Manuel avec Python

```bash
python3 main_MCAN.py --batchSize 8 --nEpochs 250 --lr 2e-3
```

Le dossier de checkpoint sera automatiquement créé avec le format :
`checkpoint_train_YYYYMMDD_HHMMSS_epochs250/`

### Méthode 3 : Spécifier un dossier personnalisé

```bash
python3 main_MCAN.py --batchSize 8 --nEpochs 250 --checkpoint_dir "mon_entrainement_special/"
```

## 📊 Continuer un entraînement existant

Si vous voulez VRAIMENT continuer un entraînement :

```bash
python3 main_MCAN.py \
    --batchSize 8 \
    --nEpochs 500 \
    --resume "checkpoint_train_20251130_160530_epochs250/De_happy_model_epoch_250.pth" \
    --start-epoch 251 \
    --checkpoint_dir "checkpoint_train_20251130_160530_epochs500_continuation/"
```

⚠️ **ATTENTION** : La continuation peut causer de l'overfitting !

## 🎯 Résumé des avantages

| Avant | Après |
|-------|-------|
| ❌ Écrasait les checkpoints | ✅ Crée un nouveau dossier unique |
| ❌ Risque de perte de modèles | ✅ Tous les modèles préservés |
| ❌ Confusion entre entraînements | ✅ Timestamp pour identifier |
| ❌ Pas de traçabilité | ✅ Logs séparés par entraînement |

## 📁 Structure des fichiers après plusieurs entraînements

```
projet_Demosaic1/
├── checkpoint/                              # Ancien (ne plus utiliser)
├── checkpoint1/                             # Modèles pré-entraînés
├── checkpoint_train_20251130_092644_epochs250/   # 1er entraînement
│   ├── De_happy_model_epoch_250.pth
│   ├── 1_train_results.csv
│   └── 1_opt.csv
├── checkpoint_train_20251130_134656_epochs300/   # 2ème entraînement
│   ├── De_happy_model_epoch_250.pth
│   ├── 1_train_results.csv
│   └── 1_opt.csv
└── checkpoint_train_20251201_103000_epochs250/   # 3ème entraînement
    ├── De_happy_model_epoch_250.pth
    ├── 1_train_results.csv
    └── 1_opt.csv
```

Chaque entraînement est **indépendant et préservé** ! 🎉

## 🔍 Identifier le meilleur modèle

Après plusieurs entraînements, consultez les fichiers CSV :

```bash
# Voir le PSNR final de chaque entraînement
tail -1 checkpoint_train_*/1_train_results.csv

# Comparer les performances
grep "^250," checkpoint_train_*/1_train_results.csv
```

## ⚠️ Notes importantes

1. **Espace disque** : Chaque checkpoint fait ~2.4 MB. Pensez à nettoyer les anciens entraînements ratés.

2. **Nommage** : Le timestamp garantit l'unicité, même si vous lancez 2 entraînements la même minute.

3. **Logs** : Tous les logs sont dans `logs/training_YYYYMMDD_HHMMSS_Nepochs.log`

4. **Meilleur modèle** : Gardez toujours le CSV d'entraînement pour retrouver les performances !
