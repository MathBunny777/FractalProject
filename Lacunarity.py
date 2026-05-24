import numpy as np
import os
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy.ndimage import uniform_filter

# Function to compute lacunarity
def compute_lacunarity(binary_image, box_size=5):
    float_img = binary_image.astype(np.float32)
    kernel_size = box_size
    local_sum = uniform_filter(float_img, size=kernel_size, mode='reflect')
    mean = np.mean(local_sum)
    variance = np.var(local_sum)
    lacunarity = (variance / (mean**2)) + 1
    return lacunarity

# Store lacunarity results
results = []

# Loop over both folders
for folderPath in ["dataset/real", "dataset/ai"]:
    label = os.path.basename(folderPath)  # 'real' or 'ai'

    for filename in tqdm(os.listdir(folderPath), desc=f"Processing {label}"):
        if filename == ".DS_Store":
            continue

        try:
            # Open and resize image
            image = Image.open(os.path.join(folderPath, filename))
            image = ImageOps.fit(image, (1024, 1024), method=Image.LANCZOS)
            gray = image.convert("L")
            gray_np = np.array(gray)

            # Binarize: white = signal
            binary = gray_np > 128

            # Compute lacunarity
            lacunarity = compute_lacunarity(binary, box_size=5)

            # Store result
            results.append((filename, label, lacunarity))

        except Exception as e:
            print(filename, "Error:", e)

# Turn into DataFrame for analysis
import pandas as pd
df = pd.DataFrame(results, columns=["Filename", "Label", "Dimension"])
df.to_csv("Lacunarity.csv", index=False)