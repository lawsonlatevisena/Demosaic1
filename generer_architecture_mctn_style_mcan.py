# -*- coding: utf-8 -*-
"""
Genere l'architecture MCTN dans le style du diagramme MCAN original
Montre clairement le remplacement de MAM par MTD
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import numpy as np

# Configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'

fig, ax = plt.subplots(figsize=(18, 6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
ax.axis('off')

# Couleurs inspirees du schema MCAN
color_orange = '#E67E22'  # MCM
color_purple = '#8E44AD'  # MRAB
color_blue = '#3498DB'    # Conv
color_red = '#C0392B'     # MAM -> MTD
color_green = '#27AE60'   # Output
color_gray = '#7F8C8D'

def draw_3d_block(x, y, width, height, depth, color, label, fontsize=11, fontweight='bold'):
    """Dessine un bloc 3D style MCAN"""
    # Face avant
    front = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color,
                          edgecolor='#2C3E50',
                          linewidth=2.5)
    ax.add_patch(front)
    
    # Face superieure (perspective)
    top_points = np.array([
        [x, y + height],
        [x + depth*0.3, y + height + depth*0.3],
        [x + width + depth*0.3, y + height + depth*0.3],
        [x + width, y + height]
    ])
    top = Polygon(top_points, facecolor=color, edgecolor='#2C3E50', 
                  linewidth=2, alpha=0.7, zorder=3)
    ax.add_patch(top)
    
    # Face droite (perspective)
    right_points = np.array([
        [x + width, y],
        [x + width + depth*0.3, y + depth*0.3],
        [x + width + depth*0.3, y + height + depth*0.3],
        [x + width, y + height]
    ])
    right = Polygon(right_points, facecolor=color, edgecolor='#2C3E50',
                   linewidth=2, alpha=0.5, zorder=2)
    ax.add_patch(right)
    
    # Texte
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center', fontsize=fontsize,
           fontweight=fontweight, color='white', zorder=4)

def draw_small_blocks_stack(x, y, width, height, colors, labels):
    """Dessine une pile de petits blocs (pour MAM/Conv)"""
    n = len(colors)
    block_width = width / n
    for i, (color, label) in enumerate(zip(colors, labels)):
        bx = x + i * block_width
        rect = Rectangle((bx, y), block_width*0.9, height,
                        facecolor=color, edgecolor='#2C3E50', linewidth=2)
        ax.add_patch(rect)
        ax.text(bx + block_width*0.45, y + height/2, label,
               ha='center', va='center', fontsize=9,
               fontweight='bold', color='white', rotation=90)

def draw_arrow(x1, y1, x2, y2, label='', style='-|>', lw=3, color='#34495E'):
    """Dessine une fleche"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color,
                           linewidth=lw,
                           zorder=1)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y - 0.3, label,
               ha='center', va='top', fontsize=9,
               style='italic', color=color)

def draw_mosaic_image(x, y, size):
    """Dessine une image mosaique style MCAN"""
    # Cadre principal
    rect = Rectangle((x, y), size, size, facecolor='white',
                    edgecolor='#2C3E50', linewidth=2.5)
    ax.add_patch(rect)
    
    # Grille de pixels coloree
    n_pixels = 8
    pixel_size = size / n_pixels
    colors_mosaic = ['#00FF00', '#0000FF', '#FF00FF', '#00FFFF', 
                    '#FFFF00', '#FF0000', '#FFA500', '#800080']
    
    for i in range(n_pixels):
        for j in range(n_pixels):
            color_idx = (i + j) % len(colors_mosaic)
            pixel = Rectangle((x + i*pixel_size, y + j*pixel_size),
                            pixel_size*0.9, pixel_size*0.9,
                            facecolor=colors_mosaic[color_idx],
                            edgecolor='none', alpha=0.7)
            ax.add_patch(pixel)

def draw_msi_cube(x, y, size):
    """Dessine le cube MSI de sortie style MCAN"""
    depth = size * 0.4
    
    # Face avant (grille bleue)
    front = Rectangle((x, y), size, size, facecolor='#3498DB',
                     edgecolor='#2C3E50', linewidth=2, alpha=0.6)
    ax.add_patch(front)
    
    # Grille
    n_lines = 10
    for i in range(n_lines):
        pos = y + i * size / n_lines
        ax.plot([x, x + size], [pos, pos], 'b-', linewidth=0.5, alpha=0.5)
        pos = x + i * size / n_lines
        ax.plot([pos, pos], [y, y + size], 'b-', linewidth=0.5, alpha=0.5)
    
    # Face superieure
    top_points = np.array([
        [x, y + size],
        [x + depth, y + size + depth],
        [x + size + depth, y + size + depth],
        [x + size, y + size]
    ])
    top = Polygon(top_points, facecolor='#5DADE2', edgecolor='#2C3E50',
                 linewidth=2, alpha=0.7)
    ax.add_patch(top)
    
    # Face droite
    right_points = np.array([
        [x + size, y],
        [x + size + depth, y + depth],
        [x + size + depth, y + size + depth],
        [x + size, y + size]
    ])
    right = Polygon(right_points, facecolor='#85C1E9', edgecolor='#2C3E50',
                   linewidth=2, alpha=0.7)
    ax.add_patch(right)

def draw_position_matrix(x, y, size):
    """Dessine la matrice de positions relatives"""
    # Cadre
    rect = Rectangle((x, y), size, size*1.2, facecolor='#ECF0F1',
                    edgecolor='#2C3E50', linewidth=2)
    ax.add_patch(rect)
    
    # Titre
    ax.text(x + size/2, y + size*1.05, 'Relative\nPosition\nMatrix',
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Matrice 4x4
    matrix_size = size * 0.7
    matrix_x = x + (size - matrix_size) / 2
    matrix_y = y + 0.1
    cell_size = matrix_size / 4
    
    for i in range(4):
        for j in range(4):
            cell = Rectangle((matrix_x + j*cell_size, matrix_y + i*cell_size),
                           cell_size*0.9, cell_size*0.9,
                           facecolor='white', edgecolor='#34495E', linewidth=1)
            ax.add_patch(cell)
            ax.text(matrix_x + j*cell_size + cell_size*0.45,
                   matrix_y + i*cell_size + cell_size*0.45,
                   f'{i},{j}', ha='center', va='center', fontsize=6)

# ===========================
# DESSIN DE L'ARCHITECTURE MCTN
# ===========================

# 1. Image mosaique d'entree
draw_mosaic_image(0.5, 2, 1.5)
ax.text(1.25, 0.8, 'Spectral Mosaic\nImage Y', ha='center', va='top',
       fontsize=10, fontweight='bold')

# Fleche
draw_arrow(2.2, 2.75, 3.0, 2.75)

# 2. MCM (Mosaic Convolution Module)
draw_3d_block(3.0, 1.8, 1.5, 1.9, 0.5, color_orange, 'MCM', fontsize=12)

# Position Matrix (au-dessus de MCM)
draw_position_matrix(3.2, 4.2, 0.8)
ax.plot([3.6, 3.75], [4.2, 3.8], 'k-', linewidth=2)

# Fleche sortie MCM
draw_arrow(4.7, 2.75, 5.4, 2.75, r'$F_{con}(\cdot)$')

# 3. MAM/Conv stack (remplace par MTD dans MCTN)
draw_small_blocks_stack(5.4, 1.8, 1.0, 1.9, 
                        [color_red, color_blue],
                        ['MTD', 'Conv'])

# Detail MTD dans un cadre en pointille
mtd_detail_x = 5.6
mtd_detail_y = 4.0
mtd_box = FancyBboxPatch((mtd_detail_x-0.3, mtd_detail_y-0.2), 2.8, 1.9,
                        boxstyle="round,pad=0.1",
                        facecolor='white',
                        edgecolor=color_red,
                        linewidth=3,
                        linestyle='--',
                        alpha=0.9)
ax.add_patch(mtd_box)

# Composants MTD
mtd_components = ['Conv', 'ReLU', 'Conv', 'MTD']
mtd_colors = ['#3498DB', '#3498DB', '#3498DB', color_red]
comp_width = 0.55
start_x = mtd_detail_x
for i, (comp, col) in enumerate(zip(mtd_components, mtd_colors)):
    cx = start_x + i * comp_width
    rect = Rectangle((cx, mtd_detail_y), comp_width*0.9, 1.3,
                    facecolor=col, edgecolor='#2C3E50', linewidth=2)
    ax.add_patch(rect)
    ax.text(cx + comp_width*0.45, mtd_detail_y + 0.65, comp,
           ha='center', va='center', fontsize=9,
           fontweight='bold', color='white', rotation=90)
    if i < len(mtd_components) - 1:
        ax.annotate('', xy=(cx + comp_width*0.9 + 0.05, mtd_detail_y + 0.65),
                   xytext=(cx + comp_width*0.9, mtd_detail_y + 0.65),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))

# Fleche vers MRAB
draw_arrow(6.6, 2.75, 7.5, 2.75)

# 4. MRAB (Mosaic Residual Attention Blocks)
mrab_positions = [7.5, 9.2, 10.9]
for i, pos in enumerate(mrab_positions):
    draw_3d_block(pos, 1.8, 1.4, 1.9, 0.5, color_purple, 'MRAB', fontsize=11)
    if i < len(mrab_positions) - 1:
        draw_arrow(pos + 1.6, 2.75, pos + 1.9, 2.75)
    elif i == 1:
        # Points de suspension
        ax.text(pos + 1.7, 2.75, '...', ha='center', va='center',
               fontsize=20, fontweight='bold', color='#34495E')

# Etiquette Hard Splitting & Interpolation
ax.text(9.4, 1.2, r'Hard Splitting & Interpolation $(H_{in}(\cdot))$',
       ha='center', va='center', fontsize=9, style='italic',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Notation au-dessus des MRAB
ax.text(9.4, 4.0, r'$H_{MRAB}(\cdot)$', ha='center', va='center',
       fontsize=11, style='italic', color=color_purple)

# Fleche sortie MRAB
draw_arrow(12.5, 2.75, 13.2, 2.75, r'$H_R(\cdot)$')

# 5. Conv finale
draw_3d_block(13.2, 2.0, 0.8, 1.5, 0.4, color_blue, 'Conv', fontsize=10)

# 6. Addition (+)
circle = Circle((14.5, 2.75), 0.3, facecolor='white',
               edgecolor='#2C3E50', linewidth=2.5, zorder=5)
ax.add_patch(circle)
ax.text(14.5, 2.75, '+', ha='center', va='center',
       fontsize=18, fontweight='bold', zorder=6)

# Connexion residuelle (long path)
ax.plot([2.2, 2.2, 14.5, 14.5], [2.2, 0.8, 0.8, 2.4],
       'k-', linewidth=3, alpha=0.6)

draw_arrow(14.1, 2.75, 14.2, 2.75)
draw_arrow(14.8, 2.75, 15.5, 2.75)

# 7. Cube MSI de sortie
draw_msi_cube(16.0, 1.8, 1.8)
ax.text(17.5, 0.8, r'MSI Cube $\hat{X}$', ha='center', va='top',
       fontsize=10, fontweight='bold')

# ===========================
# LEGENDE ET ANNOTATIONS
# ===========================

# Titre
ax.text(10, 5.7, 'MCTN Architecture (Mosaic CNN-Transformer Network)',
       ha='center', va='top', fontsize=14, fontweight='bold', color='#2C3E50')

# Note importante sur MTD
ax.text(10, 0.3, 'MTD (Mosaic Transformer Descriptor) replaces MAM with: Patch Embedding + Multi-Head Attention (4 heads) + Multi-Scale Aggregation',
       ha='center', va='center', fontsize=9, style='italic',
       color=color_red, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE6E6', 
                edgecolor=color_red, linewidth=2))

plt.tight_layout()
plt.savefig('architecture_MCTN.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('architecture_MCTN_hd.png', dpi=600, bbox_inches='tight', facecolor='white')
print("Architecture MCTN generee avec succes!")
print("   - architecture_MCTN.png (300 dpi)")
print("   - architecture_MCTN_hd.png (600 dpi)")
plt.close()
