import os
import numpy as np
import tifffile
import scipy.io as sio

# === Configuration ===
mat_dir = "CAVE_dataset/mat_files"  # Dossier contenant les .mat originaux
output_dir = "CAVE_dataset/new_val"  # Dossier de sauvegarde des .tif
os.makedirs(output_dir, exist_ok=True)

# === Indices des 16 bandes visibles (400 à 700 nm → 31 bandes totales)
# Sélection uniforme sur le spectre visible
selected_indices = np.round(np.linspace(0, 30, 16)).astype(int)  # 0 à 30 → 16 bandes

# === Traitement de chaque fichier .mat ===
for filename in os.listdir(mat_dir):
    if filename.endswith(".mat"):
        path = os.path.join(mat_dir, filename)
        try:
            data = sio.loadmat(path)
            if 'ms' not in data:
                print(f"❌ 'ms' non trouvé dans {filename}")
                continue

            ms_image = data['ms']  # [H, W, 31]
            ms_16bands = ms_image[:, :, selected_indices]  # [H, W, 16]
            ms_16bands = np.moveaxis(ms_16bands, -1, 0)  # [16, H, W]

            # Mise à l'échelle en uint16
            ms_16_uint16 = (np.clip(ms_16bands, 0, 1) * 65535).astype(np.uint16)

            # Nom de sortie
            base_name = filename.replace('_ms.mat', '_16bands.tif')
            out_path = os.path.join(output_dir, base_name)

            # Sauvegarde TIFF
            tifffile.imwrite(out_path, ms_16_uint16)
            print(f"✅ Converti : {filename} → {base_name}")

        except Exception as e:
            print(f"⚠️ Erreur pour {filename} : {e}")
