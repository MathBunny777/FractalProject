import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy.stats import linregress
import pywt  # Wavelet transform

# Wavelet FD function
def wavelet_fractal_dim(image, wavelet='db4', levels=5):
    coeffs = pywt.wavedec2(image, wavelet, level=levels)
    # Skip the approximation coefficients (coeffs[0])
    energies = [np.sum(np.abs(detail)**2) for detail in coeffs[1:]]
    scales = [2**i for i in range(1, levels+1)]
    
    slope, _, _, _, _ = linregress(np.log(scales), np.log(energies))
    return (2 + slope) / 2

results = []

# Loop over both folders
for folderPath in ["dataset/real", "dataset/ai"]:
    label = os.path.basename(folderPath)  # 'real' or 'ai'

    for filename in tqdm(os.listdir(folderPath), desc=f"Processing {label}"):
        if filename == ".DS_Store":
            continue

        try:
            # Open and resize
            image = Image.open(os.path.join(folderPath, filename))
            image = ImageOps.fit(image, (1024, 1024), method=Image.LANCZOS)
            gray = image.convert("L")
            gray_np = np.array(gray).astype(np.float32) / 255.0  # Normalize

            # Compute Wavelet FD
            fd = wavelet_fractal_dim(gray_np)

            results.append((filename, label, fd))

        except Exception as e:
            print(filename, "Error:", e)

# Save results
df = pd.DataFrame(results, columns=["Filename", "Label", "Dimension"])
df.to_csv("WaveletFractalDimension.csv", index=False)
print("✅ Wavelet FD saved to CSV")