import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

results = []

for folderPath in ["dataset/real", "dataset/ai"]:
    label = os.path.basename(folderPath)

    for filename in tqdm(os.listdir(folderPath), desc=f"SSI - {label}"):
        if filename.startswith("."):
            continue

        try:
            img = Image.open(os.path.join(folderPath, filename)).convert("L")
            img = ImageOps.fit(img, (1024, 1024))
            img_np = np.array(img)

            # Downscale and upscale
            small = Image.fromarray(img_np).resize((512, 512), Image.BILINEAR)
            upscaled = small.resize((1024, 1024), Image.BILINEAR)

            ssi = ssim(img_np, np.array(upscaled))
            results.append((filename, label, ssi))

        except Exception as e:
            print(filename, e)

df = pd.DataFrame(results, columns=["Filename", "Label", "Dimension"])
df.to_csv("SelfSimilarityIndex.csv", index=False)
print("✅ Self-Similarity Index saved")