import pandas as pd
import numpy as np
from scipy import stats

# Load CSV
df = pd.read_csv("HurstExponent.csv")

print("Columns found in CSV:", list(df.columns))

# Automatically detect value column (anything not Filename or Label)
value_col = [c for c in df.columns if c.lower() not in ["filename", "label"]][0]

print("Using value column:", value_col)

# Separate groups
real = df[df["Label"] == "real"][value_col]
ai   = df[df["Label"] == "ai"][value_col]

print("\n=== Hurst Exponent ===")

print("→ Real")
print("  Count:", len(real))
print("  Mean:", np.mean(real))
print("  Median:", np.median(real))
print("  Std Dev:", np.std(real, ddof=1))
print("  Min:", np.min(real))
print("  Max:", np.max(real))

print("\n→ AI")
print("  Count:", len(ai))
print("  Mean:", np.mean(ai))
print("  Median:", np.median(ai))
print("  Std Dev:", np.std(ai, ddof=1))
print("  Min:", np.min(ai))
print("  Max:", np.max(ai))

t_stat, p_val = stats.ttest_ind(real, ai, equal_var=False)

print("\n→ Statistical Test")
print("  t-statistic:", t_stat)
print("  p-value:", p_val)
print("  ", "❗ Significant" if p_val < 0.05 else "✅ Not significant")