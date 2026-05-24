import numpy as np  # Numerical operations
import os  # File and directory handling
from PIL import Image, ImageOps  # Image processing
from tqdm import tqdm  # Progress bars
import csv  # CSV writing
from skimage import feature

# ---------------- CONFIG ----------------
# 2026 Research suggests AI images often have higher FD due to artificial noise.
# Thresholds depend on your specific dataset:
# Typical Real: 1.2 - 1.4 | Typical AI: >1.5 or <1.1
FD_UPPER_LIMIT = 1.45 
FD_LOWER_LIMIT = 1.15
# ---------------------------------------

# Create CSV file for storing fractal dimensions
with open("FractalDimension_2026.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Label", "Dimension"])

    for folderPath in ["dataset/real", "dataset/ai"]:
        label = os.path.basename(folderPath)  # 'real' or 'ai'

        for filename in tqdm(os.listdir(folderPath), desc=f"Processing {label} images"):
            if filename == ".DS_Store":
                continue

            fullPath = os.path.join(folderPath, filename)

            try:
                # 1️⃣ Open image
                image = Image.open(fullPath)

                # 2️⃣ Resize to 1024x1024 without distortion
                image = ImageOps.fit(image, (1024, 1024), method=Image.LANCZOS)

                # 3️⃣ Convert to grayscale
                gray = image.convert("L")

                # 4️⃣ Normalize to 0–1 float array
                grayArray = np.array(gray) / 255.0

                # 6️⃣ Canny Edge Detection
                edges = feature.canny(grayArray, sigma=0.8)

                # 9️⃣ Convert to 1-bit binary image mode to prevent any anti-aliasing
                image = Image.fromarray((edges * 255).astype(np.uint8)).convert("1")

                # 🔟 Convert to boolean array
                imageArray = np.array(image) > 0

                # 1️⃣1️⃣ Box sizes
                boxSizes = [512, 256, 128, 64, 32, 16, 8, 4, 2]
                countList = []

                # 1️⃣2️⃣ Box counting loop
                for size in boxSizes:
                     # Reshape logic for vectorized counting (faster than nested loops)
                    reshaped = imageArray.reshape(1024 // size, size, 1024 // size, size)
                    count = np.sum(np.any(reshaped, axis=(1, 3)))
                    countList.append(count)

                # 1️⃣3️⃣ Calculate fractal dimension
                sizes = np.array(boxSizes)
                counts = np.array(countList)
                mask = counts > 0

                x = np.log(1 / sizes[mask])
                y = np.log(counts[mask])
                fractalDimension = np.polyfit(x, y, 1)[0]

                # 1️⃣4️⃣ Write to CSV
                writer.writerow([filename, label, fractalDimension])

            except Exception as e:
                print(filename, e)

print("✅")