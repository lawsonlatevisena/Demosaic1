#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération du graphique d'évolution du PSNR pendant l'entraînement
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration matplotlib pour un rendu propre
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# Chargement des données
print("📊 Chargement des données d'entraînement...")
df = pd.read_csv('checkpoint/1_train_results.csv')

# Création de la figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ============================================================
# Graphique 1 : Évolution du PSNR
# ============================================================
ax1.plot(df['Epoch'], df['psnr'], 'b-', linewidth=2, label='PSNR (dB)')
ax1.fill_between(df['Epoch'], df['psnr'], alpha=0.3, color='blue')

# Marqueur pour le PSNR maximum
max_psnr_idx = df['psnr'].idxmax()
max_psnr = df.loc[max_psnr_idx, 'psnr']
max_epoch = df.loc[max_psnr_idx, 'Epoch']

ax1.plot(max_epoch, max_psnr, 'r*', markersize=20, label=f'Max PSNR: {max_psnr:.2f} dB (Epoch {max_epoch})')
ax1.axhline(y=max_psnr, color='r', linestyle='--', alpha=0.3, linewidth=1)

# Configuration de l'axe
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('PSNR (dB)', fontweight='bold', color='blue')
ax1.set_title('Évolution du PSNR pendant l\'entraînement (250 epochs)', 
              fontweight='bold', fontsize=15, pad=20)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='lower right', framealpha=0.9)

# Ajout de statistiques
final_psnr = df.iloc[-1]['psnr']
initial_psnr = df.iloc[0]['psnr']
improvement = final_psnr - initial_psnr

textstr = f'PSNR Initial: {initial_psnr:.2f} dB\n'
textstr += f'PSNR Final: {final_psnr:.2f} dB\n'
textstr += f'Amélioration: +{improvement:.2f} dB'

ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================
# Graphique 2 : Évolution de la Loss
# ============================================================
ax2.plot(df['Epoch'], df['all_loss'], 'r-', linewidth=2, label='Total Loss')
ax2.fill_between(df['Epoch'], df['all_loss'], alpha=0.3, color='red')

# Marqueur pour la loss minimum
min_loss_idx = df['all_loss'].idxmin()
min_loss = df.loc[min_loss_idx, 'all_loss']
min_loss_epoch = df.loc[min_loss_idx, 'Epoch']

ax2.plot(min_loss_epoch, min_loss, 'g*', markersize=20, 
         label=f'Min Loss: {min_loss:.2f} (Epoch {min_loss_epoch})')
ax2.axhline(y=min_loss, color='g', linestyle='--', alpha=0.3, linewidth=1)

# Configuration de l'axe
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Loss', fontweight='bold', color='red')
ax2.set_title('Évolution de la Loss pendant l\'entraînement', 
              fontweight='bold', fontsize=14, pad=20)
ax2.tick_params(axis='y', labelcolor='red')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9)

# Ajout de statistiques
final_loss = df.iloc[-1]['all_loss']
initial_loss = df.iloc[0]['all_loss']
reduction = initial_loss - final_loss
reduction_pct = (reduction / initial_loss) * 100

textstr = f'Loss Initiale: {initial_loss:.2f}\n'
textstr += f'Loss Finale: {final_loss:.2f}\n'
textstr += f'Réduction: {reduction:.2f} ({reduction_pct:.1f}%)'

ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Ajustement et sauvegarde
plt.tight_layout()
output_path = 'resultats_demo/graphique_evolution_training_250epochs.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Graphique sauvegardé: {output_path}")

# ============================================================
# Graphique 3 : Vue combinée compacte
# ============================================================
fig2, ax = plt.subplots(figsize=(14, 6))

# PSNR sur l'axe principal
color = 'tab:blue'
ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('PSNR (dB)', color=color, fontweight='bold', fontsize=12)
line1 = ax.plot(df['Epoch'], df['psnr'], color=color, linewidth=2.5, label='PSNR')
ax.tick_params(axis='y', labelcolor=color)
ax.grid(True, alpha=0.3, linestyle='--')

# Loss sur l'axe secondaire
ax2_twin = ax.twinx()
color = 'tab:red'
ax2_twin.set_ylabel('Loss', color=color, fontweight='bold', fontsize=12)
line2 = ax2_twin.plot(df['Epoch'], df['all_loss'], color=color, linewidth=2.5, 
                       linestyle='--', label='Loss')
ax2_twin.tick_params(axis='y', labelcolor=color)

# Titre et légende combinée
ax.set_title('Évolution PSNR et Loss - Entraînement 250 Epochs', 
             fontweight='bold', fontsize=15, pad=20)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right', framealpha=0.9, fontsize=11)

# Marqueurs des points clés
ax.plot(max_epoch, max_psnr, 'b*', markersize=15)
ax.annotate(f'Max PSNR: {max_psnr:.2f} dB', 
            xy=(max_epoch, max_psnr), xytext=(max_epoch-30, max_psnr-2),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
output_path2 = 'resultats_demo/graphique_evolution_combine_250epochs.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Graphique combiné sauvegardé: {output_path2}")

print("\n🎉 Génération des graphiques terminée !")
print(f"\n📊 Statistiques finales:")
print(f"   • PSNR: {initial_psnr:.2f} → {final_psnr:.2f} dB (+{improvement:.2f} dB)")
print(f"   • Loss: {initial_loss:.2f} → {final_loss:.2f} (-{reduction_pct:.1f}%)")
print(f"   • PSNR Maximum: {max_psnr:.2f} dB (epoch {max_epoch})")
print(f"   • Loss Minimum: {min_loss:.2f} (epoch {min_loss_epoch})")
