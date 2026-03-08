# -*- coding: utf-8 -*-
"""
Script pour générer le diagramme de flux de données MCTN (Mosaic CNN-Transformer Network)
avec architecture détaillée incluant MTD, Pos2Weight, et reconstruction
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.axis('off')

# Couleurs pour les différents modules
color_input = '#E8F4F8'
color_mtd = '#FFE6CC'
color_pos2weight = '#D4E6F1'
color_reconstruction = '#D5F4E6'
color_output = '#F8E8E8'
color_loss = '#FFE6E6'

# Style des boîtes
box_style = "round,pad=0.1"
edge_color = '#2C3E50'
arrow_style = '-|>,head_width=0.4,head_length=0.3'

def draw_box(x, y, width, height, text, color, fontsize=9, fontweight='normal'):
    """Dessine une boîte avec texte"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=box_style,
                         facecolor=color,
                         edgecolor=edge_color,
                         linewidth=2)
    ax.add_patch(box)
    
    # Texte centré
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_offset = (len(lines) - 1 - 2*i) * 0.15
        ax.text(x + width/2, y + height/2 + y_offset, line,
               ha='center', va='center', fontsize=fontsize,
               fontweight=fontweight, color='#2C3E50')

def draw_arrow(x1, y1, x2, y2, label='', style=arrow_style, color='#34495E', lw=2):
    """Dessine une flèche avec label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color,
                           linewidth=lw,
                           zorder=1)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label,
               ha='center', va='bottom',
               fontsize=8, color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def draw_module_header(x, y, width, text, color):
    """Dessine un en-tête de module"""
    header = FancyBboxPatch((x, y), width, 0.5,
                           boxstyle="round,pad=0.05",
                           facecolor=color,
                           edgecolor=edge_color,
                           linewidth=2.5,
                           alpha=0.9)
    ax.add_patch(header)
    ax.text(x + width/2, y + 0.25, text,
           ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

# ===========================
# TITRE PRINCIPAL
# ===========================
ax.text(8, 13.3, 'MCTN: Mosaic CNN-Transformer Network',
       ha='center', va='top', fontsize=16, fontweight='bold', color='#2C3E50')
ax.text(8, 12.8, 'Flux de Données et Architecture Complète',
       ha='center', va='top', fontsize=12, color='#7F8C8D')

# ===========================
# ENTRÉE (INPUT)
# ===========================
draw_box(0.5, 11, 2, 1, 'Entrée\nMosaïque\n512×512×16', color_input, fontsize=10, fontweight='bold')
draw_arrow(2.5, 11.5, 3.5, 11.5, 'x_mosaic')

# ===========================
# MODULE MTD (Mosaic Transformer Descriptor)
# ===========================
# En-tête MTD
draw_module_header(3.5, 11.8, 5.5, 'MTD (Mosaic Transformer Descriptor)', '#E67E22')

# 1. Mosaic Patch Embedding
draw_box(3.7, 10.2, 2.3, 1.2, 'Mosaic Patch\nEmbedding\nConv 3×3\n16→64', '#FFE6CC', fontsize=8)
draw_arrow(5, 10.8, 5, 9.7, 'patches\n128×128×64', color='#E67E22')

# 2. Spatial Shuffling
draw_box(3.7, 8.5, 2.3, 1, 'Spatial Shuffling\n512×512→128×128\nRéduction 17×', '#FFE6CC', fontsize=8)
draw_arrow(5, 9.5, 5, 9.5)

# 3. Multi-Head Self-Attention
draw_box(6.5, 10.2, 2.3, 1.2, 'Multi-Head\nSelf-Attention\n4 têtes\nK, Q, V', '#FFE6CC', fontsize=8)
draw_arrow(6, 10.8, 6.5, 10.8)
draw_arrow(8.8, 10.8, 9.5, 10.8, 'attention\nmaps', color='#E67E22')

# 4. Multi-Scale Aggregation
draw_box(6.5, 8.5, 2.3, 1.2, 'Multi-Scale\nAggregation\nConv 3×3, 5×5, 7×7\nConcatenation', '#FFE6CC', fontsize=8)
draw_arrow(7.65, 9.7, 7.65, 9.0)

# Sortie MTD
draw_box(9.5, 9.5, 2, 1.5, 'Features MTD\n128×128×192\nMulti-échelle\n+ Attention', '#FFD699', fontsize=9, fontweight='bold')
draw_arrow(8.8, 9.2, 9.5, 9.2)

# ===========================
# MODULE POS2WEIGHT
# ===========================
draw_module_header(3.5, 7.3, 5.5, 'Pos2Weight (Génération de Poids Adaptatifs)', '#3498DB')

# Position Encoding
draw_box(3.7, 5.7, 2, 1.2, 'Position\nEncoding\nCoordonnées\nNormalisées', '#D4E6F1', fontsize=8)
draw_arrow(2.5, 11, 4.7, 6.9, 'positions', color='#3498DB', lw=1.5)

# MLP pour génération de poids
draw_box(6.2, 5.7, 2.8, 1.2, 'MLP (3 couches)\nFC: 2→256→256→192\nTanh + Softmax\nPoids adaptatifs', '#D4E6F1', fontsize=8)
draw_arrow(5.7, 6.3, 6.2, 6.3)

# Sortie Pos2Weight
draw_box(9.5, 5.7, 2, 1.2, 'Poids W\n192 canaux\nSpatialement\nadaptatifs', '#AED6F1', fontsize=9, fontweight='bold')
draw_arrow(9.0, 6.3, 9.5, 6.3)

# ===========================
# FUSION MTD + Pos2Weight
# ===========================
draw_box(12, 7.5, 2.5, 1.5, 'Fusion\nF = MTD ⊗ W\nProduit\nélément par élément', '#9DC3E6', fontsize=9, fontweight='bold')
draw_arrow(11.5, 10.3, 12, 8.8, 'MTD features', color='#E67E22', lw=2)
draw_arrow(11.5, 6.3, 12, 7.5, 'Poids W', color='#3498DB', lw=2)

# ===========================
# RECONSTRUCTION
# ===========================
draw_module_header(3.5, 4.3, 10.5, 'Module de Reconstruction Multi-Échelle', '#27AE60')

# Convolutions de reconstruction
draw_box(4, 2.5, 2.2, 1.3, 'Conv Block 1\n3 Conv 3×3\n192→64→64→64\nReLU + BN', '#D5F4E6', fontsize=8)
draw_box(6.5, 2.5, 2.2, 1.3, 'Conv Block 2\n3 Conv 3×3\n64→64→64→64\nReLU + BN', '#D5F4E6', fontsize=8)
draw_box(9, 2.5, 2.2, 1.3, 'Conv Block 3\n3 Conv 3×3\n64→64→64→32\nReLU + BN', '#D5F4E6', fontsize=8)
draw_box(11.5, 2.5, 2.2, 1.3, 'Conv Final\n1 Conv 3×3\n32→16\nReconstruction', '#D5F4E6', fontsize=8)

# Flèches reconstruction
draw_arrow(13.25, 7.5, 5.1, 3.8, 'features\nfusionnées', color='#27AE60')
draw_arrow(5.1, 2.5, 6.5, 2.5)
draw_arrow(6.5, 3.1, 7.6, 3.1)
draw_arrow(8.7, 2.5, 9, 2.5)
draw_arrow(9, 3.1, 10.1, 3.1)
draw_arrow(11.2, 2.5, 11.5, 2.5)
draw_arrow(11.5, 3.1, 12.6, 3.1)

# ===========================
# SORTIE
# ===========================
draw_box(13.8, 2.5, 2, 1.3, 'Image\nReconstruite\n512×512×16\nFull Resolution', color_output, fontsize=10, fontweight='bold')
draw_arrow(13.7, 3.1, 13.8, 3.1)

# ===========================
# LOSS ET SUPERVISION
# ===========================
draw_box(13.8, 0.5, 2, 1, 'Ground Truth\n512×512×16', '#E8F4F8', fontsize=9)
draw_box(11, 0.5, 2.3, 1, 'Loss Function\nL1 + SSIM\nOptimisation', color_loss, fontsize=9, fontweight='bold')

draw_arrow(13.8, 2.5, 14.8, 1.5, '', color='#E74C3C', lw=2)
draw_arrow(14.8, 1.5, 13.3, 1.0, '', color='#E74C3C', lw=2)

# ===========================
# ANNOTATIONS ET INFORMATIONS
# ===========================
# Informations clés sur le côté
info_y = 10.5
ax.text(0.2, info_y, 'Informations Cles:', fontsize=10, fontweight='bold', color='#2C3E50')
info_y -= 0.6
ax.text(0.2, info_y, '• Entree: Mosaique 512x512x16', fontsize=8, color='#34495E')
info_y -= 0.4
ax.text(0.2, info_y, '• MTD: Attention 4 tetes', fontsize=8, color='#E67E22')
info_y -= 0.4
ax.text(0.2, info_y, '• Shuffling: 17x reduction', fontsize=8, color='#E67E22')
info_y -= 0.4
ax.text(0.2, info_y, '• Pos2Weight: Poids adaptatifs', fontsize=8, color='#3498DB')
info_y -= 0.4
ax.text(0.2, info_y, '• Fusion: Produit elementxelement', fontsize=8, color='#9DC3E6')
info_y -= 0.4
ax.text(0.2, info_y, '• Reconstruction: Multi-echelle', fontsize=8, color='#27AE60')
info_y -= 0.4
ax.text(0.2, info_y, '• Loss: L1 + SSIM', fontsize=8, color='#E74C3C')
info_y -= 0.4
ax.text(0.2, info_y, '• Parametres: 616,064', fontsize=8, color='#2C3E50')
info_y -= 0.4
ax.text(0.2, info_y, '• Performance: 45.95 dB', fontsize=8, color='#27AE60', fontweight='bold')

# Légende des couleurs
legend_y = 1.8
ax.text(0.2, legend_y, 'Legende Modules:', fontsize=9, fontweight='bold', color='#2C3E50')
legend_y -= 0.35
for color, label in [
    (color_input, 'Entrée/Données'),
    (color_mtd, 'MTD (Transformer)'),
    (color_pos2weight, 'Pos2Weight'),
    (color_reconstruction, 'Reconstruction'),
    (color_output, 'Sortie'),
    (color_loss, 'Loss/Supervision')
]:
    rect = mpatches.Rectangle((0.2, legend_y-0.15), 0.3, 0.2, 
                              facecolor=color, edgecolor=edge_color, linewidth=1)
    ax.add_patch(rect)
    ax.text(0.6, legend_y, label, fontsize=7, va='center', color='#2C3E50')
    legend_y -= 0.3

# Pied de page
ax.text(8, 0.2, 'MCTN Architecture - 250 Epochs Training - PSNR: 45.95 dB, SSIM > 0.99',
       ha='center', va='bottom', fontsize=9, style='italic', color='#7F8C8D')

plt.tight_layout()
plt.savefig('flux_donnees_MCTN.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('flux_donnees_MCTN_hd.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✅ Diagramme de flux MCTN généré avec succès!")
print("   - flux_donnees_MCTN.png (300 dpi)")
print("   - flux_donnees_MCTN_hd.png (600 dpi)")
plt.close()
