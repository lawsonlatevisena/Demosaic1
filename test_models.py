#!/usr/bin/env python3
"""
Script de test simple pour les modèles MCAN
"""

import torch
import os

def test_model_loading():
    print("🔍 Test de chargement des modèles...")
    
    model_paths = [
        "checkpoint1/mcan_model.pth",
        "checkpoint1/De_happy_model_epoch_8500.pth",
        "checkpoint1/De_happy_model_epoch_8000.pth",
        "checkpoint1/De_happy_model_epoch_7500.pth"
    ]
    
    for model_path in model_paths:
        print(f"\n📁 Test: {model_path}")
        if not os.path.exists(model_path):
            print("   ❌ Fichier inexistant")
            continue
            
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            print(f"   ✅ Chargement réussi")
            print(f"   📝 Type: {type(checkpoint)}")
            
            if hasattr(checkpoint, '__class__'):
                print(f"   🏷️  Classe: {checkpoint.__class__}")
                
            if isinstance(checkpoint, dict):
                print(f"   🔑 Clés: {list(checkpoint.keys())}")
                
            # Test si c'est un modèle Net
            if hasattr(checkpoint, 'forward'):
                print("   🎯 C'est un modèle PyTorch avec forward()")
                print("   💡 Suggestion: Utiliser directement ce modèle")
                
                # Test simple
                try:
                    checkpoint.eval()
                    print("   ✅ Modèle mis en mode évaluation")
                    return checkpoint, model_path
                except Exception as e:
                    print(f"   ⚠️  Problème d'évaluation: {e}")
                    
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    return None, None

if __name__ == "__main__":
    model, path = test_model_loading()
    if model:
        print(f"\n🎉 Modèle utilisable trouvé: {path}")
        print("💡 Vous pouvez maintenant l'utiliser pour l'inférence!")
    else:
        print("\n❌ Aucun modèle utilisable trouvé")