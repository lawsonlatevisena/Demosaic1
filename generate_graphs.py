#!/usr/bin/env python3
"""
Générateur de graphiques pour visualiser les métriques PSNR et SSIM par bande
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
RESULTS_DIR = "resultats_demo"
METRICS_CSV = os.path.join(RESULTS_DIR, "metriques_16_bandes.csv")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "graphique_metriques.png")

# Charger les métriques
df = pd.read_csv(METRICS_CSV)

# Créer une figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('📊 Évaluation des Métriques par Bande Spectrale - MCTN', 
             fontsize=16, fontweight='bold', y=0.995)

bands = df['Bande'].values
psnr = df['PSNR_dB'].values
ssim = df['SSIM'].values

# Graphique 1: PSNR
colors_psnr = plt.cm.viridis(np.linspace(0.3, 0.9, len(bands)))
bars1 = ax1.bar(bands, psnr, color=colors_psnr, edgecolor='black', linewidth=1.2, alpha=0.8)

# Ligne de moyenne PSNR
psnr_mean = psnr.mean()
ax1.axhline(y=psnr_mean, color='red', linestyle='--', linewidth=2, 
            label=f'Moyenne: {psnr_mean:.2f} dB')

# Ajouter les valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars1, psnr)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Bande Spectrale', fontsize=12, fontweight='bold')
ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
ax1.set_title('Peak Signal-to-Noise Ratio (PSNR) par Bande', fontsize=13, pad=10)
ax1.set_xticks(bands)
ax1.set_ylim([35, 40])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10)

# Graphique 2: SSIM
colors_ssim = plt.cm.plasma(np.linspace(0.3, 0.9, len(bands)))
bars2 = ax2.bar(bands, ssim, color=colors_ssim, edgecolor='black', linewidth=1.2, alpha=0.8)

# Ligne de moyenne SSIM
ssim_mean = ssim.mean()
ax2.axhline(y=ssim_mean, color='red', linestyle='--', linewidth=2, 
            label=f'Moyenne: {ssim_mean:.4f}')

# Ajouter les valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars2, ssim)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Bande Spectrale', fontsize=12, fontweight='bold')
ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
ax2.set_title('Structural Similarity Index (SSIM) par Bande', fontsize=13, pad=10)
ax2.set_xticks(bands)
ax2.set_ylim([0.985, 1.0])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"✅ Graphique sauvegardé: {OUTPUT_PNG}")
print(f"📊 Statistiques:")
print(f"   PSNR - Moyenne: {psnr_mean:.2f} dB, Min: {psnr.min():.2f} dB, Max: {psnr.max():.2f} dB")
print(f"   SSIM - Moyenne: {ssim_mean:.4f}, Min: {ssim.min():.4f}, Max: {ssim.max():.4f}")

# Créer un graphique combiné avec courbes
fig2, ax3 = plt.subplots(figsize=(14, 8))

# Créer deux axes Y
color1 = 'tab:blue'
ax3.set_xlabel('Bande Spectrale', fontsize=12, fontweight='bold')
ax3.set_ylabel('PSNR (dB)', color=color1, fontsize=12, fontweight='bold')
line1 = ax3.plot(bands, psnr, color=color1, marker='o', linewidth=2.5, 
                 markersize=8, label='PSNR', linestyle='-', markeredgecolor='black')
ax3.tick_params(axis='y', labelcolor=color1)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_ylim([35, 40])

# Deuxième axe Y pour SSIM
ax4 = ax3.twinx()
color2 = 'tab:red'
ax4.set_ylabel('SSIM', color=color2, fontsize=12, fontweight='bold')
line2 = ax4.plot(bands, ssim, color=color2, marker='s', linewidth=2.5, 
                 markersize=8, label='SSIM', linestyle='-', markeredgecolor='black')
ax4.tick_params(axis='y', labelcolor=color2)
ax4.set_ylim([0.985, 1.0])

# Titre et légende
ax3.set_title('📈 Évolution des Métriques PSNR et SSIM par Bande Spectrale', 
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(bands)

# Combiner les légendes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
output_combined = os.path.join(RESULTS_DIR, "graphique_metriques_combine.png")
plt.savefig(output_combined, dpi=300, bbox_inches='tight')
print(f"✅ Graphique combiné sauvegardé: {output_combined}")

print(f"\n🎨 Pour visualiser:")
print(f"   xdg-open {OUTPUT_PNG}")
print(f"   xdg-open {output_combined}")

plt.close('all')
