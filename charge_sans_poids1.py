import os
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from lapsrn import Net
from My_function import input_matrix_wpn

# === Configuration
msfa_size = 4
image_path = "CAVE_dataset/new_val/fake_and_real_peppers_ms_IMECMine_HA.tif"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialiser le modèle (non entraîné)
model = Net().to(device)
model.eval()

# === Fonction de chargement TIFF
def load_tiff(path):
    img = tifffile.imread(path)
    if img.ndim == 3:
        if img.shape[0] == 16:
            img = img[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
        elif img.shape[2] == 16:
            img = np.moveaxis(img, 2, 0)[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
    raw_tensor = torch.from_numpy(img.copy()).float().unsqueeze(0) / 65535.0
    input_tensor = raw_tensor[:, [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
    return raw_tensor, input_tensor

# === Charger et préparer l'image
raw_tensor, input_tensor = load_tiff(image_path)
H, W = raw_tensor.shape[2], raw_tensor.shape[3]
scale_map = input_matrix_wpn(H, W, msfa_size).to(device)

# === Inférence (sans poids)
with torch.no_grad():
    output = model([input_tensor.to(device), raw_tensor.to(device)], scale_map)
output = output.squeeze().cpu().numpy()
output = np.clip(output, 0, 1)

# === Création du dossier de sortie
output_dir = "resultats_bandes"
os.makedirs(output_dir, exist_ok=True)

# === 1. Sauvegarder chaque bande individuellement en PNG
for i in range(16):
    bande_path = os.path.join(output_dir, f"bande_{i}.png")
    plt.imsave(bande_path, output[i], cmap='gray')
    print(f"✅ Enregistré : {bande_path}")

# === 2. Sauvegarder l'image multibande au format TIFF
tif_path = os.path.join(output_dir, "sortie_multibande.tif")
output_uint16 = (output * 65535).astype(np.uint16)
tifffile.imwrite(tif_path, output_uint16)
print(f"✅ TIFF multibande : {tif_path}")

# === 3. Générer et sauvegarder image RGB simulée
rgb_bandes = [8, 4, 2]  # R, G, B
rgb_image = np.stack([output[i] for i in rgb_bandes], axis=-1)
rgb_image = np.clip(rgb_image, 0, 1)
rgb_path = os.path.join(output_dir, "visualisation_RGB.png")
plt.imsave(rgb_path, rgb_image)
print(f"✅ Image RGB simulée : {rgb_path}")
