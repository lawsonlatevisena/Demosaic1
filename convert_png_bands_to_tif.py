import os
import numpy as np
import tifffile
import imageio.v2 as imageio  # Utilisé pour lire les PNG

# === Configuration ===
input_root = "CAVE_dataset/png_bands"  # Dossier contenant les sous-dossiers par scène
output_dir = "CAVE_dataset/new_val"
os.makedirs(output_dir, exist_ok=True)

# === Sélection de 16 bandes uniformément réparties parmi 31
selected_indices = np.round(np.linspace(0, 30, 16)).astype(int)  # ex: [0, 2, 4, ..., 30]

# === Parcours de chaque sous-dossier
for scene_name in os.listdir(input_root):
    scene_path = os.path.join(input_root, scene_name)
    if not os.path.isdir(scene_path):
        continue

    png_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.png')])
    if len(png_files) != 31:
        print(f"⚠️ {scene_name} ignoré : {len(png_files)} bandes trouvées (attendu : 31)")
        continue

    try:
        # Charger les 31 bandes
        bands = []
        for fname in png_files:
            img_path = os.path.join(scene_path, fname)
            img = imageio.imread(img_path)
            bands.append(img)

        bands = np.stack(bands, axis=0)  # [31, H, W]

        # Sélectionner 16 bandes
        bands_16 = bands[selected_indices, :, :]  # [16, H, W]
        bands_16 = (np.clip(bands_16 / 255.0, 0, 1) * 65535).astype(np.uint16)

        # Sauvegarde en .tif
        output_path = os.path.join(output_dir, f"{scene_name}_16bands.tif")
        tifffile.imwrite(output_path, bands_16)
        print(f"✅ {scene_name} → {output_path}")

    except Exception as e:
        print(f"❌ Erreur avec {scene_name} : {e}")
