#!/usr/bin/env python3
"""
Script simplifié pour tester le modèle MCAN et générer des images démosaïquées
"""

import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from libtiff import TIFFfile
from lapsrn import Net
from My_function import reorder_imec, input_matrix_wpn

def load_img(filepath):
    """Charger une image TIFF"""
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    return img

def mask_input(GT_image):
    """Appliquer le masque MSFA"""
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 16), dtype=np.float32)
    mask[0::4, 0::4, 0] = 1
    mask[0::4, 1::4, 1] = 1
    mask[0::4, 2::4, 2] = 1
    mask[0::4, 3::4, 3] = 1
    mask[1::4, 0::4, 4] = 1
    mask[1::4, 1::4, 5] = 1
    mask[1::4, 2::4, 6] = 1
    mask[1::4, 3::4, 7] = 1
    mask[2::4, 0::4, 8] = 1
    mask[2::4, 1::4, 9] = 1
    mask[2::4, 2::4, 10] = 1
    mask[2::4, 3::4, 11] = 1
    mask[3::4, 0::4, 12] = 1
    mask[3::4, 1::4, 13] = 1
    mask[3::4, 2::4, 14] = 1
    mask[3::4, 3::4, 15] = 1
    input_image = mask * GT_image
    return input_image

def main():
    print("🚀 Test du modèle MCAN...")
    
    # Chargement du modèle
    try:
        print("📋 Chargement du modèle...")
        model = Net()
        
        # Essayer différents modèles
        model_paths = [
            "checkpoint1/mcan_model.pth",
            "checkpoint1/De_happy_model_epoch_8500.pth",
            "checkpoint1/De_happy_model_epoch_8000.pth"
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                print(f"   Tentative avec {model_path}...")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Vérifier le type de checkpoint  
                if hasattr(checkpoint, 'state_dict'):
                    # C'est un objet modèle PyTorch
                    model = checkpoint
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    model.load_state_dict(state_dict, strict=False)
                elif isinstance(checkpoint, dict):
                    # C'est un state_dict direct
                    model.load_state_dict(checkpoint, strict=False)
                else:
                    # C'est probablement l'objet modèle complet
                    model = checkpoint
                print(f"   ✅ Modèle chargé avec succès: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"   ❌ Échec: {e}")
                continue
        
        if not model_loaded:
            print("❌ Impossible de charger aucun modèle")
            return
            
        model.eval()
        
        # Test avec une image
        print("🖼️  Chargement d'une image de test...")
        image_path = "CAVE_dataset/new_val/beads_ms_IMECMine_D65.tif"
        
        try:
            im_gt_y = load_img(image_path)
            print(f"   Image chargée: {im_gt_y.shape}")
            
            # Normalisation
            max_val = np.max(im_gt_y)
            im_gt_y = im_gt_y / max_val * 255
            im_gt_y = im_gt_y.transpose(1, 0, 2)
            
            # Application du masque
            im_l_y = mask_input(im_gt_y)
            im_l_y = reorder_imec(im_l_y)
            im_gt_y = reorder_imec(im_gt_y)
            
            # Préparation pour le modèle
            im_input = im_l_y.astype(float) / 255.
            im_input = im_input.transpose(2, 0, 1)
            raw = im_input.sum(axis=0)
            
            # Conversion en tensors
            im_input_tensor = torch.from_numpy(im_input).float().unsqueeze(0)
            raw_tensor = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0)
            scale_coord_map = input_matrix_wpn(raw.shape[0], raw.shape[1])
            
            print("🔄 Inférence...")
            with torch.no_grad():
                try:
                    output = model([im_input_tensor, raw_tensor], scale_coord_map)
                    im_output = output.data[0].numpy().astype(np.float32)
                    
                    # Post-traitement
                    im_output = im_output * 255.
                    im_output = np.clip(im_output, 0, 255).astype(np.uint8)
                    
                    print("💾 Sauvegarde des résultats...")
                    
                    # Sauvegarder quelques bandes
                    for i in range(min(3, im_output.shape[0])):
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(im_gt_y[:, :, i], cmap='gray')
                        plt.title(f'Ground Truth - Bande {i+1}')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(im_l_y[:, :, i], cmap='gray')
                        plt.title(f'Input Mosaïqué - Bande {i+1}')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(im_output[i, :, :], cmap='gray')
                        plt.title(f'MCAN Output - Bande {i+1}')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(f'resultat_bande_{i+1}.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"   Sauvegardé: resultat_bande_{i+1}.png")
                    
                    # Sauvegarder le cube complet
                    tifffile.imwrite('resultat_complet_16bandes.tif', 
                                   im_output.transpose(1, 2, 0).astype(np.uint8))
                    print("   Sauvegardé: resultat_complet_16bandes.tif")
                    
                    print("✅ Test terminé avec succès!")
                    print("🎯 Images générées:")
                    print("   - resultat_bande_1.png, resultat_bande_2.png, resultat_bande_3.png")
                    print("   - resultat_complet_16bandes.tif")
                    
                except Exception as e:
                    print(f"❌ Erreur lors de l'inférence: {e}")
                    
        except Exception as e:
            print(f"❌ Erreur lors du chargement de l'image: {e}")
            
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()