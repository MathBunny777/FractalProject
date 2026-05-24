import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import nolds

results = []

for folderPath in ["dataset/real", "dataset/ai"]:
    label = os.path.basename(folderPath)

    for filename in tqdm(os.listdir(folderPath), desc=f"Hurst - {label}"):
        if filename.startswith("."):
            continue

        try:
            img = Image.open(os.path.join(folderPath, filename)).convert("L")
            img = ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)
            arr = np.array(img) / 255.0

            signal = arr.mean(axis=0)  # 1D texture signal
            H = nolds.hurst_rs(signal)

            results.append((filename, label, H))

        except Exception as e:
            print(filename, e)

df = pd.DataFrame(results, columns=["Filename", "Label", "Dimension"])
df.to_csv("HurstExponent.csv", index=False)
print("✅ Hurst Exponent saved")