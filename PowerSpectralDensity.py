import numpy as np
import pandas as pd
import os
import csv
from PIL import Image, ImageOps
from scipy import fftpack
from tqdm import tqdm

# ---------------- CONFIG ----------------
# AI Detection in 2026: Look for peaks that deviate from the 1/f slope
PEAK_THRESHOLD = 2.5 # Sensitivity to spectral spikes
# ---------------------------------------

def compute_psd_slope(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift) ** 2

    # FIXED: Direct unpacking from shape
    cy, cx = psd.shape[0] // 2, psd.shape[1] // 2
    
    y, x = np.indices(psd.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())

    radial_psd = tbin / np.maximum(nr, 1)
    radial_psd = radial_psd[1:]

    log_r = np.log(np.arange(1, len(radial_psd)+1))
    log_p = np.log(radial_psd + 1e-8)

    # FIXED: Cleaner attribute access
    result = linregress(log_r, log_p)
    return -result.slope
    
def compute_radial_profile(psd_2d):
    """Compute the 1D radial average of a 2D PSD."""
    y, x = np.indices(psd_2d.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(int)
    
    # Sum power at each radius
    tbin = np.bincount(r.ravel(), psd_2d.ravel())
    # Count pixels at each radius
    nr = np.bincount(r.ravel())
    
    radial_profile = tbin / (nr + 1e-12)
    return radial_profile


results = []

with open("PowerSpectralDensity.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["File Name", "Label", "Dimension"])

    for folderPath in ["dataset/real", "dataset/ai"]:
        label = os.path.basename(folderPath)

        min_range = 1000
        max_range = -1000

        for filename in tqdm(os.listdir(folderPath), desc=f"PSD - {label}"):
            if filename.startswith("."):
                continue

            try:
                img = Image.open(os.path.join(folderPath, filename))
                img = ImageOps.fit(img, (1024, 1024))
                arr = np.array(img) / 255.0

                gray = np.array(img.convert("L")) / 255.0

                # 2. Compute 2D Fast Fourier Transform
                f_transform = fftpack.fft2(gray)
                f_shifted = fftpack.fftshift(f_transform)
            
                # 3. Compute Power Spectrum Density (Magnitude Squared)
                psd_2d = np.abs(f_shifted)**2
                
                # 4. Collapse to 1D Radial Profile
                psd_1d = compute_radial_profile(psd_2d)
                
                # 5. DETECT AI PEAKS (2026 Logic)
                # On a log-scale, natural images are a straight line. 
                # We calculate the deviation from the local mean to find "spikes".
                log_psd = np.log10(psd_1d + 1e-12)
                
                # Simple high-pass filter on the 1D signal to find spikes
                # Subtracting moving average to isolate peaks
                moving_avg = np.convolve(log_psd, np.ones(5)/5, mode='same')
                spikes = log_psd - moving_avg
                
                # Spectral Spike Score: Max deviation in high-frequency range
                # We ignore the very center (low freq) and very edges
                spike_score = np.max(spikes[50:200]) 

                if spike_score > max_range:
                    max_range = spike_score
                if spike_score < min_range:
                    min_range = spike_score
                
                results.append((filename, label, spike_score))

                # slope = compute_psd_slope(arr)
                # results.append((filename, label, slope))
            except Exception as e:
                print(filename, e)
                
        # print("min & max", min_range, max_range)

df = pd.DataFrame(results, columns=["Filename", "Label", "Dimension"])
df.to_csv("PowerSpectralDensity.csv", index=False)
print("✅")