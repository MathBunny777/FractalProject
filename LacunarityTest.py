import numpy as np  # Numerical operations
import os  # File and directory handling
from PIL import Image, ImageOps  # Image processing
from tqdm import tqdm  # Progress bars
import csv  # CSV writing
from scipy.signal import convolve2d

# Create CSV file for storing fractal dimensions
with open("Lacunarity.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["File Name", "Label", "Fractal Dimension"])

    # for folderPath in ["dataset/real", "dataset/ai"]:
    for folderPath in ["datatest"]:
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

                # 9️⃣ Convert to 1-bit binary image mode to prevent any anti-aliasing
                image = Image.fromarray((grayArray * 255).astype(np.uint8)).convert("1")

                # 🔟 Convert to boolean array
                imageArray = np.array(image) > 0

                # 1️⃣1️⃣ Box sizes
                boxSizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
                lacunarityList = []

                # 1️⃣2️⃣ Box counting loop
                for size in boxSizes:
                    count = 0
                    for i in range(0, 1024, size):
                        for j in range(0, 1024, size):
                            box = imageArray[i:i+size, j:j+size]
                            count += np.sum(box)
                    countList.append(count)

                    # 1️⃣3️⃣ Calculate Lacnuarity
                    mu = np.mean(countList)
                    sigma = np.std(countList)

                    if mu == 0:
                        lacunarity = 0
                    else:
                        lacunarity = (sigma / mu) ** 2
                    lacunarityList.append(lacunarity)

                # 1️⃣4️⃣ Write to CSV
                writer.writerow([filename, label, lacunarity])

            except Exception as e:
                print(filename, e)

print("✅")