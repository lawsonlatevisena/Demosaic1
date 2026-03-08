#!/usr/bin/env python3
"""
Générateur de galerie HTML pour visualiser les résultats de démosaïquage
"""
import os
import pandas as pd
from datetime import datetime

# Configuration
RESULTS_DIR = "resultats_demo"
OUTPUT_HTML = os.path.join(RESULTS_DIR, "galerie_resultats.html")
METRICS_CSV = os.path.join(RESULTS_DIR, "metriques_16_bandes.csv")

# Charger les métriques
df = pd.read_csv(METRICS_CSV)

# Générer le HTML
html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Galerie de Résultats - MCTN Démosaïquage</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid #667eea;
        }}
        
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .stats-overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .metrics-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .metrics-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .metrics-table tr:hover {{
            background: #f8f9ff;
        }}
        
        .quality-excellent {{
            color: #10b981;
            font-weight: bold;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }}
        
        .gallery-item {{
            background: #f8f9ff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .gallery-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}
        
        .gallery-item img {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .band-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .metrics {{
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }}
        
        .metric {{
            text-align: center;
            flex: 1;
        }}
        
        .metric-label {{
            font-size: 0.85em;
            color: #666;
        }}
        
        .metric-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        
        .composite-section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9ff;
            border-radius: 10px;
        }}
        
        .composite-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        
        .composite-image {{
            max-width: 800px;
            margin: 0 auto;
            display: block;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        
        footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #eee;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎨 Galerie de Résultats - MCTN</h1>
            <p class="subtitle">Démosaïquage d'Images Multispectrales (16 Bandes)</p>
            <p class="subtitle">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
        </header>
        
        <section class="stats-overview">
            <div class="stat-card">
                <div class="stat-label">PSNR Moyen</div>
                <div class="stat-value">{df['PSNR_dB'].mean():.2f} dB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SSIM Moyen</div>
                <div class="stat-value">{df['SSIM'].mean():.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">PSNR Max</div>
                <div class="stat-value">{df['PSNR_dB'].max():.2f} dB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SSIM Max</div>
                <div class="stat-value">{df['SSIM'].max():.3f}</div>
            </div>
        </section>
        
        <h2 style="color: #667eea; margin-bottom: 20px;">📊 Métriques par Bande</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Bande</th>
                    <th>PSNR (dB)</th>
                    <th>SSIM</th>
                    <th>Qualité</th>
                </tr>
            </thead>
            <tbody>
"""

# Ajouter les lignes du tableau
for _, row in df.iterrows():
    html_content += f"""                <tr>
                    <td><strong>Bande {int(row['Bande'])}</strong></td>
                    <td>{row['PSNR_dB']:.2f}</td>
                    <td>{row['SSIM']:.3f}</td>
                    <td class="quality-excellent">{row['Qualite']}</td>
                </tr>
"""

html_content += """            </tbody>
        </table>
        
        <div class="composite-section">
            <h2>🌈 Composite RGB</h2>
            <img src="comparaison_rgb_composite.png" alt="Composite RGB" class="composite-image">
        </div>
        
        <h2 style="color: #667eea; margin: 40px 0 20px 0;">🔬 Comparaisons par Bande Spectrale</h2>
        <div class="gallery">
"""

# Ajouter les images de comparaison
for _, row in df.iterrows():
    band_num = int(row['Bande'])
    psnr = row['PSNR_dB']
    ssim = row['SSIM']
    
    html_content += f"""            <div class="gallery-item">
                <div class="band-title">Bande {band_num}</div>
                <img src="comparaison_bande_{band_num}.png" alt="Comparaison Bande {band_num}">
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">PSNR</div>
                        <div class="metric-value">{psnr:.2f} dB</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">SSIM</div>
                        <div class="metric-value">{ssim:.3f}</div>
                    </div>
                </div>
            </div>
"""

html_content += f"""        </div>
        
        <footer>
            <p><strong>MCTN - Mosaic Convolution Attention Network</strong></p>
            <p>Projet de démosaïquage d'images hyperspectrales</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                📁 Fichiers source: {RESULTS_DIR}/ | 
                📊 {len(df)} bandes spectrales analysées
            </p>
        </footer>
    </div>
</body>
</html>
"""

# Sauvegarder le fichier HTML
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Galerie HTML générée: {OUTPUT_HTML}")
print(f"📊 {len(df)} bandes spectrales incluses")
print(f"🌐 Pour visualiser: xdg-open {OUTPUT_HTML}")
