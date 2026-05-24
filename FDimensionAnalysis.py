import pandas as pd
import re
from typing import NamedTuple

class Data(NamedTuple):
    filename: str
    typename: str
    realIsHigher: bool

data = [
    Data("FractalDimension.csv", "Fractal Dimension", True),
    Data("FractalDimension_2026.csv", "FractalDimension_2026", True),
    Data("HurstExponent.csv", "HurstExponent", True),
    Data("Lacunarity.csv", "Lacunarity", False),
#    Data("PowerSpectralDensity.csv", "Power Spectral Density", True),
    Data("SelfSimilarityIndex.csv", "Self Similarity Index", True),
    Data("WaveletFractalDimension.csv", "Wavelet Fractal Dimension", True),
]

for d in data:
    df = pd.read_csv(d.filename)
    name = d.typename
    realIsHigher = d.realIsHigher

    # === 2. Fixed column names (based on your CSV) ===
    filename_col = "Filename"
    fd_col = "Dimension"

    # === 3. Clean base image ID ===
    def get_base_id(name):
        name = name.lower()
        name = re.sub(r'(_real|_ai)', '', name)
        name = re.sub(r'\.\w+$', '', name)
        return name

    df["BaseID"] = df[filename_col].apply(get_base_id)

    # === 4. Split REAL and AI ===
    real_df = df[df["Label"].str.lower() == "real"].copy()
    ai_df   = df[df["Label"].str.lower() == "ai"].copy()

    # === 5. Merge REAL–AI pairs ===
    paired = pd.merge(
        real_df,
        ai_df,
        on="BaseID",
        suffixes=("_real", "_ai")
    )

    if len(paired) == 0:
        raise ValueError("❌ No real–AI pairs found.")

    # === 6. Classification rule (REAL > AI) ===
    realHigh = paired[f"{fd_col}_real"] > paired[f"{fd_col}_ai"]
    aiHigh = paired[f"{fd_col}_ai"] > paired[f"{fd_col}_real"]

    paired["Correct"] = realHigh if realIsHigher else aiHigh
    paired["Difference"] = paired[f"{fd_col}_real"] - paired[f"{fd_col}_ai"]

    accuracy = paired["Correct"].mean() * 100

    # === 7. Results ===
    print("\n📊 " + name + " Classification")
    print(f"Image pairs analyzed: {len(paired)}")
    print(f"Average (Real − AI): {paired['Difference'].mean():.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    # === 8. Save output ===
    saveFile = "ClassificationResults_" + name + ".csv"
    paired.to_csv(saveFile, index=False)
    print("💾 Results saved to: " + saveFile)
