import numpy as np  # Numerical operations
import os  # File and directory handling
import csv  # CSV writing
from PIL import Image, ImageOps  # Image processing
from tqdm import tqdm  # Progress bars
from scipy.stats import linregress  # Linear regression

# ---------------- CONFIG ----------------
q_values = np.linspace(-5, 5, 21)
box_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
EPS = 1e-12
THRESHOLD = 0.15
# 2026 Detection Thresholds (Tune based on your initial results)
# Real images typically have Width > 0.8 and Asymmetry > 0.05
WIDTH_THRESHOLD = 0.85
ASYM_THRESHOLD = 0.03
# ---------------------------------------

with open("MultifractalSpectrum.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(["File Name", "Label", "Predicted AI"] + [f"tau(q={q:.1f})" for q in q_values])
    writer.writerow(["File Name", "Label", "Curvature Score", "Predicted AI"] + [f"tau(q={q:.1f})" for q in q_values])

    for folderPath in ["dataset/ai", "dataset/real"]:
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

                # 5️⃣ Precompute box masses μ for each scale
                scale_masses = {}

                for size in box_sizes:
                    # transforms that flat 2D inmage into a 4-dimensional array (a "grid of blocks")
                    # Dimension 1 & 3 (1024 // size): These represent the number of blocks vertically and horizontally.
                    # Dimension 2 & 4 (size): These represent the actual height and width of each individual block. 
                    # If size = 32, the code reshapes the 1024x1024 array into a (32, 32, 32, 32) shape, This creates a 32x32 grid of small tiles, Each tile is 32x32 pixels in size
                    reshaped = grayArray.reshape(
                        1024 // size, size,
                        1024 // size, size
                    )

                    # looks at the 4D array, which is structured as (number_of_blocks_vertically, block_height, number_of_blocks_horizontally, block_width)
                    # By summing across axis 1 and 3, it collapses the "inner" dimensions (the pixel values within each block).
                    # The result is a 2D grid where each cell contains the sum of all pixels in that corresponding block.
                    # flatten -> It takes that 2D grid of sums and converts it into a 1D array -> useful if you need to iterate through the block sums one by one or feed them into a standard statistical function
                    mu_values = reshaped.sum(axis=(1, 3)).flatten()

                    # This variable now holds the resulting list of sums. 
                    mu_values = mu_values[mu_values > 0]  # remove empty boxes

                    # np.sum(mu_values) -> total sum of every element in the mu_values array -> In image terms, this represents the total brightness or energy of the entire original 1024x1024 image.
                    # + EPS (Epsilon) to prevent division by zero (if the sum is 0)
                    # The resulting normalized array is stored in a dictionary called scale_masses, using the current size (e.g., 32 or 64) as the key. 
                    scale_masses[size] = mu_values / (np.sum(mu_values) + EPS)  # ✅ probabilities

                # 6️⃣ Compute τ(q)
                tau_values = []
                log_sizes = np.log(1 / np.array(box_sizes))

                for q in q_values:
                    Zq = []
                    for size in box_sizes:
                        mu = scale_masses[size]
                        Zq.append(np.sum(mu ** q) + EPS)

                    log_Zq = np.log(Zq)
                    slope, _, _, _, _ = linregress(log_sizes, log_Zq)
                    tau_values.append(slope)

                # 3. SINGULARITY SPECTRUM CALCULATION (2026 Logic)
#                tau_array = np.array(tau_values)
                
                # alpha(q) = d(tau)/dq
#                alpha = np.gradient(tau_array, q_values)
                
                # f(alpha) = q * alpha - tau(q)
#                f_alpha = q_values * alpha - tau_array

                # 4. METRIC EXTRACTION
                # Width (Delta Alpha): Higher in real images, lower in AI
#                width = np.max(alpha) - np.min(alpha)
                
                # Asymmetry: Real images are skewed; AI is often too symmetric
#                mid = len(alpha) // 2
#                left_side = np.abs(alpha[mid] - alpha[0])
#                right_side = np.abs(alpha[-1] - alpha[mid])
#                asymmetry = np.abs(left_side - right_side)

                # 5. DECISION
                # Flag as AI if the spectrum is suspiciously narrow or too perfectly balanced
#                is_ai = "Yes" if (width < WIDTH_THRESHOLD or asymmetry < ASYM_THRESHOLD) else "No"



                # --- 2026 DETECTION ENGINE ---
                # Calculate second derivative (curvature)
                # We normalize by the step size to get the true mathematical curvature
                dq = q_values[1] - q_values[0]
                second_deriv = np.gradient(np.gradient(tau_values, dq), dq)

                # We focus on absolute curvature. 
                # AI = "Hyper-textured" (High Curvature)
                # Real = "Naturally balanced" (Medium Curvature)
                total_curvature = np.mean(np.abs(second_deriv))

                # If it's too curved (overly complex), it's AI. 
                # If it's too flat (monofractal/simple), it's also likely AI.
                # Real photos typically sit in a "Golden Zone."
                # is_ai = "Yes" if (total_curvature > THRESHOLD or total_curvature < 0.05) else "No"

                # 7️⃣ Write to CSV
                # writer.writerow([filename, label, is_ai] + tau_values)
                writer.writerow([filename, label, total_curvature])

            except Exception as e:
                print(filename, e)

print("✅")