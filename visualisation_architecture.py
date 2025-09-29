#!/usr/bin/env python3
"""
Visualisation de l'architecture du modèle MCAN
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_mcan_architecture_diagram():
    """Créer un diagramme détaillé de l'architecture MCAN"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Définir les couleurs pour différents types de blocs
    colors = {
        'input': '#E8F4FD',
        'denoising': '#FFE6E6',
        'wb_conv': '#E6F3E6',
        'p2w': '#FFF2E6',
        'attention': '#F0E6FF',
        'conv_block': '#E6F7FF',
        'output': '#F0F8E6',
        'connection': '#DDDDDD'
    }
    
    # Fonction pour créer des boîtes avec du texte
    def create_box(x, y, width, height, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='black',
                           linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                wrap=True)
    
    # Fonction pour créer des flèches
    def create_arrow(x1, y1, x2, y2, color='black', style='->', lw=2):
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle=style, shrinkA=5, shrinkB=5,
                              color=color, lw=lw)
        ax.add_patch(arrow)
    
    # Titre principal
    ax.text(10, 13.5, 'Architecture MCAN (Mosaic Convolution-Attention Network)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # === ENTRÉES ===
    create_box(0.5, 11.5, 2.5, 1, 'Image Mosaïquée\n(x)\n16 × 512 × 512', colors['input'], 8)
    create_box(0.5, 10, 2.5, 1, 'Image Raw\n(y)\n1 × 512 × 512', colors['input'], 8)
    create_box(0.5, 8.5, 2.5, 1, 'Position Matrix\n(pos_mat)\n262144 × 2', colors['input'], 8)
    
    # === BRANCHE PRINCIPALE (Image Mosaïquée) ===
    
    # Débruitage
    create_box(4, 11.5, 2.5, 1, 'Denoising Conv\n16→64\n3×3, pad=1', colors['denoising'], 8)
    create_box(7, 11.5, 2.5, 1, 'LeakyReLU\n(0.2)', colors['denoising'], 8)
    create_box(10, 11.5, 2.5, 1, 'Denoising Output\n64→16\n3×3, pad=1', colors['denoising'], 8)
    
    # Soustraction du bruit
    create_box(13, 11.5, 2, 1, 'x - noise\n(Denoised)', colors['denoising'], 8)
    
    # WB Conv
    create_box(13, 10, 2.5, 1, 'WB Conv\n16→16\n7×7, groups=16', colors['wb_conv'], 8)
    
    # === BRANCHE P2W (Position to Weight) ===
    create_box(4, 8.5, 2.5, 1, 'P2W Network\nPos→Weight\n2→400', colors['p2w'], 8)
    create_box(7, 8.5, 2.5, 1, 'Local Weight\nGeneration', colors['p2w'], 8)
    
    # === TRAITEMENT RAW ===
    create_box(4, 10, 2.5, 1, 'Repeat Y\n(Up-sampling)', colors['conv_block'], 8)
    create_box(7, 10, 2.5, 1, 'Unfold\n(5×5 patches)', colors['conv_block'], 8)
    create_box(10, 10, 2.5, 1, 'MatMul avec\nLocal Weights', colors['conv_block'], 8)
    create_box(10, 8.5, 2.5, 1, 'Raw Conv\nOutput', colors['conv_block'], 8)
    
    # === RÉSEAU PRINCIPAL ===
    
    # Branch Front
    create_box(16.5, 10, 2.5, 1, 'Branch Front\nMA Layer\n(16 channels)', colors['attention'], 8)
    
    # Front Conv Input
    create_box(16.5, 8.5, 2.5, 1, 'Front Conv\n16→64\n3×3, pad=1', colors['conv_block'], 8)
    
    # Attention Blocks F1 et F2
    create_box(16.5, 7, 2.5, 1, 'Conv-Attention\nBlock F1\n(64 channels)', colors['attention'], 8)
    create_box(16.5, 5.5, 2.5, 1, 'Conv-Attention\nBlock F2\n(64 channels)', colors['attention'], 8)
    
    # Branch Back
    create_box(16.5, 4, 2.5, 1, 'Branch Back\n64→16\n3×3, pad=1', colors['conv_block'], 8)
    
    # === SORTIE FINALE ===
    create_box(13, 4, 2.5, 1, 'Addition\nHR_4x + WB', colors['output'], 8)
    create_box(13, 2.5, 2.5, 1, 'Image Finale\n16 × 512 × 512', colors['output'], 8)
    
    # === DÉTAILS DES BLOCS ATTENTION ===
    
    # Détail MA Layer
    create_box(0.5, 5.5, 3, 2, 'MA Layer Detail:\n• Shuffle Down (4×)\n• Linear Projection\n• Multi-Head Attention\n• FC + Sigmoid\n• Shuffle Up (4×)', colors['attention'], 7)
    
    # Détail Conv-Attention Block
    create_box(0.5, 3, 3, 2, 'Conv-Attention Block:\n• 3× Conv2d (3×3)\n• LeakyReLU\n• MA Layer\n• Residual Connection', colors['attention'], 7)
    
    # === FLÈCHES DE CONNEXION ===
    
    # Entrées vers traitement
    create_arrow(3, 12, 4, 12)  # Image mosaïquée vers débruitage
    create_arrow(3, 10.5, 4, 10.5)  # Raw vers repeat
    create_arrow(3, 9, 4, 9)  # Position vers P2W
    
    # Chaîne de débruitage
    create_arrow(6.5, 12, 7, 12)
    create_arrow(9.5, 12, 10, 12)
    create_arrow(12.5, 12, 13, 12)
    
    # Vers WB Conv
    create_arrow(14, 11.5, 14, 11)
    
    # Chaîne Raw
    create_arrow(6.5, 10.5, 7, 10.5)
    create_arrow(9.5, 10.5, 10, 10.5)
    create_arrow(11.25, 10, 11.25, 9.5)
    
    # P2W vers Raw Conv
    create_arrow(6.5, 9, 10, 9)
    
    # Raw Conv vers Branch Front
    create_arrow(12.5, 9, 16.5, 9)
    create_arrow(17.75, 9, 17.75, 9.5)
    
    # Chaîne principale
    create_arrow(17.75, 9.5, 17.75, 8.5)
    create_arrow(17.75, 8.5, 17.75, 8)
    create_arrow(17.75, 7, 17.75, 6.5)
    create_arrow(17.75, 5.5, 17.75, 5)
    
    # Vers sortie
    create_arrow(16.5, 4.5, 15.5, 4.5)
    
    # WB vers addition
    create_arrow(14.25, 10, 14.25, 5)
    
    # Addition vers sortie finale
    create_arrow(14.25, 4, 14.25, 3.5)
    
    # === LÉGENDE ===
    legend_y = 1.5
    create_box(0.5, legend_y, 1, 0.5, 'Input', colors['input'], 7)
    create_box(2, legend_y, 1, 0.5, 'Denoising', colors['denoising'], 7)
    create_box(3.5, legend_y, 1, 0.5, 'WB Conv', colors['wb_conv'], 7)
    create_box(5, legend_y, 1, 0.5, 'P2W', colors['p2w'], 7)
    create_box(6.5, legend_y, 1, 0.5, 'Attention', colors['attention'], 7)
    create_box(8, legend_y, 1, 0.5, 'Conv Block', colors['conv_block'], 7)
    create_box(9.5, legend_y, 1, 0.5, 'Output', colors['output'], 7)
    
    ax.text(5, 0.8, 'Légende des Modules', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # === INFORMATIONS TECHNIQUES ===
    tech_info = """
    Caractéristiques Techniques:
    • Entrée: Image mosaïquée 16-bandes (512×512)
    • Architecture: Encoder-Decoder avec Attention
    • Modules clés: MA Layer, P2W Network, WB Conv
    • Mécanisme: Position-aware convolution weights
    • Sortie: Image démosaïquée 16-bandes (512×512)
    """
    
    ax.text(17.5, 1.5, tech_info, ha='left', va='top', 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('architecture_MCAN.png', dpi=300, bbox_inches='tight')
    plt.savefig('architecture_MCAN.pdf', bbox_inches='tight')
    
    return fig

def create_detailed_flow_diagram():
    """Créer un diagramme de flux détaillé"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Titre
    ax.text(8, 11.5, 'Flux de Données MCAN - Vue Détaillée', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Définir les positions et connexions
    stages = [
        {'name': 'Input\nMosaic Image\n(16, 512, 512)', 'pos': (1, 10), 'color': '#E8F4FD'},
        {'name': 'Denoising\nBranch', 'pos': (4, 10), 'color': '#FFE6E6'},
        {'name': 'WB\nConvolution', 'pos': (7, 10), 'color': '#E6F3E6'},
        {'name': 'MA Layer\n(Attention)', 'pos': (10, 10), 'color': '#F0E6FF'},
        {'name': 'Conv-Attention\nBlocks (F1, F2)', 'pos': (13, 10), 'color': '#F0E6FF'},
        
        {'name': 'Raw Image\n(1, 512, 512)', 'pos': (1, 7), 'color': '#E8F4FD'},
        {'name': 'Position Matrix\n(N, 2)', 'pos': (1, 4), 'color': '#E8F4FD'},
        {'name': 'P2W Network\n(Pos→Weight)', 'pos': (4, 4), 'color': '#FFF2E6'},
        {'name': 'Local Weight\nGeneration', 'pos': (7, 4), 'color': '#FFF2E6'},
        {'name': 'Mosaic\nConvolution', 'pos': (10, 7), 'color': '#E6F7FF'},
        
        {'name': 'Feature\nFusion', 'pos': (13, 7), 'color': '#E6F7FF'},
        {'name': 'Final\nAddition', 'pos': (10, 1), 'color': '#F0F8E6'},
        {'name': 'Output\nDemosaiced\n(16, 512, 512)', 'pos': (13, 1), 'color': '#F0F8E6'},
    ]
    
    # Dessiner les boîtes
    for stage in stages:
        x, y = stage['pos']
        rect = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1,
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'],
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, stage['name'], ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    # Dessiner les connexions
    connections = [
        ((1, 10), (4, 10)),  # Input to Denoising
        ((4, 10), (7, 10)),  # Denoising to WB
        ((7, 10), (10, 10)), # WB to MA Layer
        ((10, 10), (13, 10)), # MA to Conv-Attention
        
        ((1, 7), (10, 7)),   # Raw to Mosaic Conv
        ((1, 4), (4, 4)),    # Pos to P2W
        ((4, 4), (7, 4)),    # P2W to Local Weight
        ((7, 4), (10, 7)),   # Local Weight to Mosaic Conv
        
        ((10, 7), (13, 7)),  # Mosaic Conv to Fusion
        ((13, 10), (13, 7)), # Conv-Attention to Fusion
        ((13, 7), (10, 1)),  # Fusion to Final Addition
        ((7, 10), (10, 1)),  # WB to Final Addition (skip connection)
        ((10, 1), (13, 1)),  # Final Addition to Output
    ]
    
    for (x1, y1), (x2, y2) in connections:
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle='->', shrinkA=35, shrinkB=35,
                              color='black', lw=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('flux_donnees_MCAN.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_attention_mechanism_detail():
    """Créer un diagramme détaillé du mécanisme d'attention MA Layer"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Titre
    ax.text(7, 9.5, 'MA Layer (Multi-Head Attention) - Détail', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Étapes du MA Layer
    steps = [
        {'name': 'Input\n(C, H, W)', 'pos': (1, 8), 'size': (1.5, 0.8), 'color': '#E8F4FD'},
        {'name': 'Shuffle Down\n(4×)', 'pos': (3.5, 8), 'size': (1.5, 0.8), 'color': '#FFE6E6'},
        {'name': 'Flattened\n(C×16, H/4×W/4)', 'pos': (6, 8), 'size': (1.8, 0.8), 'color': '#FFE6E6'},
        {'name': 'Linear\nProjection', 'pos': (9, 8), 'size': (1.5, 0.8), 'color': '#FFF2E6'},
        {'name': 'Multi-Head\nAttention', 'pos': (11.5, 8), 'size': (1.8, 0.8), 'color': '#F0E6FF'},
        
        {'name': 'Attention\nWeights', 'pos': (11.5, 6), 'size': (1.8, 0.8), 'color': '#F0E6FF'},
        {'name': 'FC + Sigmoid\nActivation', 'pos': (9, 6), 'size': (1.8, 0.8), 'color': '#FFF2E6'},
        {'name': 'Attention Map\n(C×16, 1, 1)', 'pos': (6, 6), 'size': (1.8, 0.8), 'color': '#FFF2E6'},
        {'name': 'Element-wise\nMultiplication', 'pos': (3.5, 6), 'size': (1.8, 0.8), 'color': '#E6F7FF'},
        
        {'name': 'Shuffle Up\n(4×)', 'pos': (3.5, 4), 'size': (1.5, 0.8), 'color': '#E6F3E6'},
        {'name': 'Output\n(C, H, W)', 'pos': (6, 4), 'size': (1.5, 0.8), 'color': '#F0F8E6'},
    ]
    
    # Dessiner les étapes
    for step in steps:
        x, y = step['pos']
        w, h = step['size']
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=step['color'],
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, step['name'], ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    # Connexions
    connections = [
        ((1, 8), (3.5, 8)),
        ((3.5, 8), (6, 8)),
        ((6, 8), (9, 8)),
        ((9, 8), (11.5, 8)),
        ((11.5, 8), (11.5, 6)),
        ((11.5, 6), (9, 6)),
        ((9, 6), (6, 6)),
        ((6, 6), (3.5, 6)),
        ((6, 8), (3.5, 6)),  # Skip connection pour multiplication
        ((3.5, 6), (3.5, 4)),
        ((3.5, 4), (6, 4)),
    ]
    
    for (x1, y1), (x2, y2) in connections:
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle='->', shrinkA=15, shrinkB=15,
                              color='blue', lw=1.5)
        ax.add_patch(arrow)
    
    # Informations techniques
    tech_info = """
    Paramètres MA Layer:
    • Shuffle Down: Réduction 4× (512²→128²)
    • Channels: 16 → 256 (×16 expansion)
    • Attention Heads: 4
    • Reduction Ratio: 16
    • Activation: Sigmoid pour attention weights
    • Shuffle Up: Restauration 4× (128²→512²)
    """
    
    ax.text(1, 2.5, tech_info, ha='left', va='top', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('attention_mechanism_MCAN.png', dpi=300, bbox_inches='tight')
    
    return fig

def main():
    """Fonction principale pour générer tous les diagrammes"""
    print("🎨 Génération des diagrammes d'architecture MCAN...")
    
    # 1. Architecture générale
    print("   📊 Diagramme d'architecture générale...")
    fig1 = create_mcan_architecture_diagram()
    
    # 2. Flux de données
    print("   🔄 Diagramme de flux de données...")
    fig2 = create_detailed_flow_diagram()
    
    # 3. Mécanisme d'attention
    print("   🧠 Détail du mécanisme d'attention...")
    fig3 = create_attention_mechanism_detail()
    
    print("\n✅ Diagrammes générés:")
    print("   • architecture_MCAN.png - Architecture complète")
    print("   • architecture_MCAN.pdf - Version PDF")
    print("   • flux_donnees_MCAN.png - Flux de données")
    print("   • attention_mechanism_MCAN.png - Mécanisme d'attention")
    
    plt.show()

if __name__ == "__main__":
    main()