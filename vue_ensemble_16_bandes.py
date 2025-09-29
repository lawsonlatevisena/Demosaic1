#!/usr/bin/env python3
"""
Script pour créer une vue d'ensemble de toutes les 16 bandes dans une seule image
"""

import torch
import numpy as np
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

def create_16_bands_overview():
    print("🚀 Génération de la vue d'ensemble 16 bandes")
    print("=" * 50)
    
    # Chargement du modèle et traitement (même code que demo_final.py)
    print("📋 Chargement du modèle...")
    checkpoint_path = "checkpoint1/De_happy_model_epoch_8500.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.eval()
    print(f"   ✅ Modèle chargé")
    
    print("🖼️  Chargement et traitement de l'image...")
    image_path = "CAVE_dataset/new_val/beads_ms_IMECMine_D65.tif"
    im_gt_y = load_img(image_path)
    
    # Traitement
    max_val = np.max(im_gt_y)
    im_gt_y = im_gt_y / max_val * 255
    im_gt_y = im_gt_y.transpose(1, 0, 2)
    
    im_l_y = mask_input(im_gt_y)
    im_l_y = reorder_imec(im_l_y)
    im_gt_y = reorder_imec(im_gt_y)
    
    # Inférence
    print("🔄 Inférence...")
    im_input = im_l_y.astype(float) / 255.
    im_input = im_input.transpose(2, 0, 1)
    raw = im_input.sum(axis=0)
    
    im_input_tensor = torch.from_numpy(im_input).float().unsqueeze(0)
    raw_tensor = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0)
    scale_coord_map = input_matrix_wpn(raw.shape[0], raw.shape[1], 4)
    
    with torch.no_grad():
        output = model([im_input_tensor, raw_tensor], scale_coord_map)
        im_output = output.data[0].numpy().astype(np.float32)
        im_output = im_output * 255.
        im_output = np.clip(im_output, 0, 255).astype(np.uint8)
    
    # Créer la vue d'ensemble 4x4 pour les 16 bandes
    print("🎨 Création de la vue d'ensemble 4x4...")
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('Vue d\'ensemble des 16 Bandes Spectrales - MCAN Démosaïquage', fontsize=24, y=0.98)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Afficher la bande démosaïquée
        ax.imshow(im_output[i, :, :], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Bande {i+1}', fontsize=16, pad=10)
        ax.axis('off')
        
        # Ajouter une bordure colorée pour différencier les bandes
        colors = plt.cm.tab20(i/16)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors)
            spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('resultats_demo/vue_ensemble_16_bandes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Créer aussi une comparaison GT vs Output pour toutes les bandes
    print("🎨 Création de la comparaison complète GT vs Output...")
    
    fig, axes = plt.subplots(8, 4, figsize=(20, 40))
    fig.suptitle('Comparaison Ground Truth vs MCAN - 16 Bandes', fontsize=24, y=0.99)
    
    for i in range(16):
        # Ground Truth
        row_gt = (i * 2) // 4
        col_gt = (i * 2) % 4
        ax_gt = axes[row_gt, col_gt]
        ax_gt.imshow(im_gt_y[:, :, i], cmap='gray', vmin=0, vmax=255)
        ax_gt.set_title(f'GT Bande {i+1}', fontsize=14)
        ax_gt.axis('off')
        
        # MCAN Output
        row_out = (i * 2 + 1) // 4
        col_out = (i * 2 + 1) % 4
        ax_out = axes[row_out, col_out]
        ax_out.imshow(im_output[i, :, :], cmap='gray', vmin=0, vmax=255)
        ax_out.set_title(f'MCAN Bande {i+1}', fontsize=14)
        ax_out.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    plt.savefig('resultats_demo/comparaison_complete_gt_vs_mcan.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Vues d'ensemble créées:")
    print("   • vue_ensemble_16_bandes.png")
    print("   • comparaison_complete_gt_vs_mcan.png")

if __name__ == "__main__":
    create_16_bands_overview()