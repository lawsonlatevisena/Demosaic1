#!/usr/bin/env python3
"""
Script fonctionnel pour tester MCAN et générer des images démosaïquées
"""

import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from libtiff import TIFFfile
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
    print("🚀 MCAN - Génération d'Images Démosaïquées")
    print("=" * 50)
    
    # 1. Chargement du modèle
    print("📋 Chargement du modèle...")
    try:
        # Utiliser le modèle epoch qui contient l'objet complet
        checkpoint_path = "checkpoint1/De_happy_model_epoch_8500.pth"
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Extraire le modèle de la clé 'model'
        model = checkpoint['model']
        model.eval()
        print(f"   ✅ Modèle chargé: {checkpoint_path}")
        print(f"   📊 Époque: {checkpoint['epoch']}")
        
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return
    
    # 2. Chargement de l'image de test
    print("\n🖼️  Chargement de l'image de test...")
    try:
        image_path = "CAVE_dataset/new_val/beads_ms_IMECMine_D65.tif"
        im_gt_y = load_img(image_path)
        print(f"   ✅ Image chargée: {im_gt_y.shape}")
        
        # Normalisation
        max_val = np.max(im_gt_y)
        im_gt_y = im_gt_y / max_val * 255
        im_gt_y = im_gt_y.transpose(1, 0, 2)
        
        # Application du masque MSFA
        print("   🔄 Application du masque MSFA...")
        im_l_y = mask_input(im_gt_y)
        im_l_y = reorder_imec(im_l_y)
        im_gt_y = reorder_imec(im_gt_y)
        
    except Exception as e:
        print(f"❌ Erreur chargement image: {e}")
        return
    
    # 3. Préparation pour l'inférence
    print("\n⚙️  Préparation pour l'inférence...")
    try:
        im_input = im_l_y.astype(float) / 255.
        im_input = im_input.transpose(2, 0, 1)
        raw = im_input.sum(axis=0)
        
        # Conversion en tensors
        im_input_tensor = torch.from_numpy(im_input).float().unsqueeze(0)
        raw_tensor = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0)
        scale_coord_map = input_matrix_wpn(raw.shape[0], raw.shape[1], 4)
        
        print(f"   📐 Taille entrée: {im_input_tensor.shape}")
        print(f"   📐 Taille raw: {raw_tensor.shape}")
        
    except Exception as e:
        print(f"❌ Erreur préparation: {e}")
        return
    
    # 4. Inférence
    print("\n🔄 Inférence MCAN...")
    try:
        with torch.no_grad():
            output = model([im_input_tensor, raw_tensor], scale_coord_map)
            im_output = output.data[0].numpy().astype(np.float32)
            
            # Post-traitement
            im_output = im_output * 255.
            im_output = np.clip(im_output, 0, 255).astype(np.uint8)
            
        print(f"   ✅ Inférence terminée: {im_output.shape}")
        
    except Exception as e:
        print(f"❌ Erreur inférence: {e}")
        return
    
    # 5. Sauvegarde des résultats
    print("\n💾 Sauvegarde des résultats...")
    try:
        # Créer un dossier de résultats
        import os
        os.makedirs("resultats_demo", exist_ok=True)
        
        # Sauvegarder le cube complet 16 bandes
        tifffile.imwrite('resultats_demo/image_demosaiquee_16_bandes.tif', 
                       im_output.transpose(1, 2, 0).astype(np.uint8))
        print("   ✅ Sauvegardé: resultats_demo/image_demosaiquee_16_bandes.tif")
        
        # Sauvegarder TOUTES les 16 bandes spectrales avec comparaisons
        print("   📊 Génération des 16 bandes spectrales...")
        for i in range(16):  # Toutes les 16 bandes
            plt.figure(figsize=(18, 6))
            
            # Ground Truth
            plt.subplot(1, 3, 1)
            plt.imshow(im_gt_y[:, :, i], cmap='gray', vmin=0, vmax=255)
            plt.title(f'Ground Truth - Bande {i+1}/16', fontsize=14)
            plt.axis('off')
            
            # Input Mosaïqué
            plt.subplot(1, 3, 2)
            plt.imshow(im_l_y[:, :, i], cmap='gray', vmin=0, vmax=255)
            plt.title(f'Input Mosaïqué - Bande {i+1}/16', fontsize=14)
            plt.axis('off')
            
            # Output MCAN+MTD
            plt.subplot(1, 3, 3)
            plt.imshow(im_output[i, :, :], cmap='gray', vmin=0, vmax=255)
            plt.title(f'MCAN+MTD Démosaïqué - Bande {i+1}/16', fontsize=14)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'resultats_demo/comparaison_bande_{i+1}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Sauvegardé: resultats_demo/comparaison_bande_{i+1}.png")
        
        # Créer une image RGB composite (bandes 10, 5, 1 par exemple)
        try:
            rgb_bands = [9, 4, 0]  # Index 0-based pour les bandes 10, 5, 1
            rgb_gt = np.stack([im_gt_y[:, :, i] for i in rgb_bands], axis=2)
            rgb_output = np.stack([im_output[i, :, :] for i in rgb_bands], axis=2)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_gt.astype(np.uint8))
            plt.title('Ground Truth (RGB Composite)', fontsize=14)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(rgb_output.astype(np.uint8))
            plt.title('MCAN Démosaïqué (RGB Composite)', fontsize=14)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('resultats_demo/comparaison_rgb_composite.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Sauvegardé: resultats_demo/comparaison_rgb_composite.png")
            
        except Exception as e:
            print(f"   ⚠️  Composite RGB non créé: {e}")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return
    
    # 6. Résumé final
    print("\n" + "=" * 50)
    print("🎉 SUCCÈS ! Images démosaïquées générées")
    print("=" * 50)
    print("📁 Fichiers générés dans 'resultats_demo/':")
    print("   • image_demosaiquee_16_bandes.tif (cube complet)")
    print("   • comparaison_bande_1.png à comparaison_bande_16.png (toutes les bandes)")
    print("   • comparaison_rgb_composite.png")
    print("\n💡 Vous pouvez maintenant visualiser les résultats !")

if __name__ == "__main__":
    main()