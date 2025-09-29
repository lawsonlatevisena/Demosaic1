#!/usr/bin/env python3
"""
Script d'évaluation quantitative pour toutes les bandes spectrales
"""

import torch
import numpy as np
import tifffile
from libtiff import TIFFfile
from My_function import reorder_imec, input_matrix_wpn
from sklearn.metrics import mean_squared_error
import math

def load_img(filepath):
    """Charger une image TIFF"""
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    return img

def mask_input(GT_image):
    """Appliquer le masque MSFA"""
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 16), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            mask[i::4, j::4, i*4+j] = 1
    input_image = mask * GT_image
    return input_image

def calculate_psnr(x_true, x_pred):
    """Calculer PSNR pour chaque bande"""
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    
    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])
        x_pred_k = x_pred[:, :, k].reshape([-1])
        
        mse = mean_squared_error(x_true_k, x_pred_k)
        max_val = np.max(x_true_k)
        
        if max_val != 0 and mse != 0:
            PSNR[k] = 10 * math.log10(math.pow(max_val, 2) / mse)
        else:
            PSNR[k] = 100  # Perfect reconstruction
    
    return PSNR

def calculate_ssim_simple(x_true, x_pred):
    """Calculer SSIM simplifié pour chaque bande"""
    n_bands = x_true.shape[2]
    SSIM = np.zeros(n_bands)
    
    c1 = 0.0001
    c2 = 0.0009
    
    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])
        x_pred_k = x_pred[:, :, k].reshape([-1])
        
        mu1 = np.mean(x_true_k)
        mu2 = np.mean(x_pred_k)
        
        sigma1_sq = np.var(x_true_k)
        sigma2_sq = np.var(x_pred_k)
        sigma12 = np.cov(x_true_k, x_pred_k)[0, 1]
        
        ssim_val = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1*mu1 + mu2*mu2 + c1) * (sigma1_sq + sigma2_sq + c2))
        SSIM[k] = ssim_val
    
    return SSIM

def main():
    print("📊 ÉVALUATION QUANTITATIVE - 16 BANDES SPECTRALES")
    print("=" * 60)
    
    # 1. Chargement du modèle
    print("📋 Chargement du modèle...")
    checkpoint_path = "checkpoint1/De_happy_model_epoch_8500.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.eval()
    
    # 2. Chargement de l'image de test
    print("🖼️  Chargement de l'image de test...")
    image_path = "CAVE_dataset/new_val/beads_ms_IMECMine_D65.tif"
    im_gt_y = load_img(image_path)
    
    # Normalisation
    max_val = np.max(im_gt_y)
    im_gt_y = im_gt_y / max_val * 255
    im_gt_y = im_gt_y.transpose(1, 0, 2)
    
    # Application du masque
    im_l_y = mask_input(im_gt_y)
    im_l_y = reorder_imec(im_l_y)
    im_gt_y = reorder_imec(im_gt_y)
    
    # 3. Inférence
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
        im_output = np.clip(im_output, 0, 255)
    
    # 4. Calcul des métriques
    print("📊 Calcul des métriques pour les 16 bandes...")
    
    # Reformatage pour les calculs
    gt_formatted = im_gt_y.astype(float)
    pred_formatted = im_output.transpose(1, 2, 0).astype(float)
    
    # PSNR par bande
    psnr_bands = calculate_psnr(gt_formatted, pred_formatted)
    
    # SSIM par bande  
    ssim_bands = calculate_ssim_simple(gt_formatted, pred_formatted)
    
    # 5. Affichage des résultats
    print("\n" + "=" * 60)
    print("📈 RÉSULTATS D'ÉVALUATION")
    print("=" * 60)
    
    print(f"{'Bande':<6} {'PSNR (dB)':<12} {'SSIM':<12} {'Qualité':<15}")
    print("-" * 50)
    
    for i in range(16):
        if psnr_bands[i] > 35:
            quality = "Excellente"
        elif psnr_bands[i] > 30:
            quality = "Très bonne"
        elif psnr_bands[i] > 25:
            quality = "Bonne"
        elif psnr_bands[i] > 20:
            quality = "Moyenne"
        else:
            quality = "Faible"
            
        print(f"{i+1:<6} {psnr_bands[i]:<12.2f} {ssim_bands[i]:<12.3f} {quality:<15}")
    
    # Statistiques globales
    print("\n" + "=" * 60)
    print("📊 STATISTIQUES GLOBALES")
    print("=" * 60)
    print(f"PSNR Moyen    : {np.mean(psnr_bands):.2f} dB")
    print(f"PSNR Médian   : {np.median(psnr_bands):.2f} dB")
    print(f"PSNR Min      : {np.min(psnr_bands):.2f} dB (Bande {np.argmin(psnr_bands)+1})")
    print(f"PSNR Max      : {np.max(psnr_bands):.2f} dB (Bande {np.argmax(psnr_bands)+1})")
    print()
    print(f"SSIM Moyen    : {np.mean(ssim_bands):.3f}")
    print(f"SSIM Médian   : {np.median(ssim_bands):.3f}")
    print(f"SSIM Min      : {np.min(ssim_bands):.3f} (Bande {np.argmin(ssim_bands)+1})")
    print(f"SSIM Max      : {np.max(ssim_bands):.3f} (Bande {np.argmax(ssim_bands)+1})")
    
    # Sauvegarde des résultats
    print(f"\n💾 Sauvegarde des métriques...")
    import os
    os.makedirs("resultats_demo", exist_ok=True)
    
    # Sauvegarde en CSV
    with open("resultats_demo/metriques_16_bandes.csv", "w") as f:
        f.write("Bande,PSNR_dB,SSIM,Qualite\n")
        for i in range(16):
            quality = "Excellente" if psnr_bands[i] > 35 else "Tres_bonne" if psnr_bands[i] > 30 else "Bonne" if psnr_bands[i] > 25 else "Moyenne" if psnr_bands[i] > 20 else "Faible"
            f.write(f"{i+1},{psnr_bands[i]:.2f},{ssim_bands[i]:.3f},{quality}\n")
    
    print("   ✅ Sauvegardé: resultats_demo/metriques_16_bandes.csv")
    
    print("\n🎉 Évaluation terminée !")

if __name__ == "__main__":
    main()