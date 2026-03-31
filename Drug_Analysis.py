################################################################################
# PYTHON SCRIPT - Drug Overdose Death Rate Analysis
# U.S. Drug Overdose Mortality (1999-2018)
# Author: Kankanit Charoenpol
# Converted from R | Tools: pandas, scipy, matplotlib, seaborn
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

################################################################################
# SECTION 1: DATA IMPORT AND CLEANING
################################################################################

df = pd.read_csv("Drug_Overdose_Rate.csv")

# Convert ESTIMATE to numeric (non-numeric values become NaN)
df["ESTIMATE"] = pd.to_numeric(df["ESTIMATE"], errors="coerce")

# Remove suppressed data rows (flagged with *)
overdose_clean = df[df["FLAG"].isna() | (df["FLAG"] == "") | (df["FLAG"] != "*")].copy()

# Extract SEX from STUB_LABEL
def extract_sex(label):
    label = str(label).lower()
    if "female" in label:
        return "Female"
    elif "male" in label:
        return "Male"
    return "All"

# Extract RACE_ETHNICITY from STUB_LABEL
def extract_race(label):
    label = str(label)
    if "Hispanic" in label and "Non-Hispanic" not in label:
        return "Hispanic"
    elif "White" in label and "Hispanic" not in label:
        return "Non-Hispanic White"
    elif ("Black" in label or "African American" in label) and "Hispanic" not in label:
        return "Non-Hispanic Black"
    elif ("Asian" in label or "Pacific Islander" in label) and "Hispanic" not in label:
        return "Asian/Pacific Islander"
    elif ("American Indian" in label or "Alaska Native" in label) and "Hispanic" not in label:
        return "American Indian/Alaska Native"
    return "All Races"

overdose_clean["SEX"] = overdose_clean["STUB_LABEL"].apply(extract_sex)
overdose_clean["RACE_ETHNICITY"] = overdose_clean["STUB_LABEL"].apply(extract_race)

print("Dataset loaded and cleaned.")
print(f"Total observations: {len(overdose_clean)}\n")

################################################################################
# SECTION 2: RESEARCH QUESTION 1 - ONE-SAMPLE T-TEST
################################################################################

print("=" * 50)
print("RESEARCH QUESTION 1")
print("One-Sample T-Test: 2017 Death Rate vs CDC Benchmark (21.7)")
print("=" * 50)

data_2017 = overdose_clean[
    (overdose_clean["YEAR"] == 2017) & (overdose_clean["ESTIMATE"].notna())
]["ESTIMATE"]

# Descriptive statistics
print("\nDescriptive Statistics for 2017:")
print(data_2017.describe().round(3))

# Hypothesis test
cdc_benchmark = 21.7
t_stat, p_value = stats.ttest_1samp(data_2017, cdc_benchmark)

print(f"\nH0: μ_2017 = 21.7 | H1: μ_2017 ≠ 21.7 | α = 0.05")
print(f"t-statistic : {t_stat:.3f}")
print(f"p-value     : {p_value:.4e}")
print(f"Decision    : {'REJECT H0' if p_value < 0.05 else 'FAIL TO REJECT H0'}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data_2017, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
ax.axvline(data_2017.mean(), color="blue", linestyle="--", linewidth=1.5,
           label=f"Sample Mean = {data_2017.mean():.2f}")
ax.axvline(cdc_benchmark, color="red", linestyle="-", linewidth=1.5,
           label=f"CDC Benchmark = {cdc_benchmark}")
ax.set_title("2017 Death Rates: Sample vs CDC Benchmark", fontsize=14, fontweight="bold")
ax.set_xlabel("Death Rate per 100,000")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("Q1_2017_vs_CDC_Benchmark.png", dpi=300)
plt.show()

################################################################################
# SECTION 3: RESEARCH QUESTION 2 - TWO-SAMPLE T-TEST
################################################################################

print("\n" + "=" * 50)
print("RESEARCH QUESTION 2")
print("Two-Sample T-Test: Black Population Death Rates Over Time")
print("=" * 50)

black_data = overdose_clean[
    (overdose_clean["RACE_ETHNICITY"] == "Non-Hispanic Black") &
    (overdose_clean["ESTIMATE"].notna())
].copy()

black_data["Period"] = black_data["YEAR"].apply(lambda y:
    "Early Period (1999-2010)" if 1999 <= y <= 2010 else
    "Late Period (2014-2018)"  if 2014 <= y <= 2018 else None
)
black_data = black_data[black_data["Period"].notna()]

# Descriptive statistics by period
print("\nDescriptive Statistics by Period:")
print(black_data.groupby("Period")["ESTIMATE"].describe().round(3))

# Hypothesis test
early = black_data[black_data["Period"] == "Early Period (1999-2010)"]["ESTIMATE"]
late  = black_data[black_data["Period"] == "Late Period (2014-2018)"]["ESTIMATE"]

t_stat2, p_value2 = stats.ttest_ind(late, early, equal_var=False)

print(f"\nH0: μ_early = μ_late | H1: μ_late > μ_early | α = 0.05")
print(f"t-statistic : {t_stat2:.3f}")
print(f"p-value     : {p_value2:.4e}")
print(f"Decision    : {'REJECT H0' if p_value2 < 0.05 else 'FAIL TO REJECT H0'}")

# Visualization - Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
black_data.boxplot(column="ESTIMATE", by="Period", ax=ax, patch_artist=True)
ax.set_title("Non-Hispanic Black Death Rates: Early vs Late Period", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Death Rate per 100,000")
plt.suptitle("")
plt.tight_layout()
plt.savefig("Q2_Black_Boxplot.png", dpi=300)
plt.show()

################################################################################
# SECTION 4: RESEARCH QUESTION 3 - TWO-SAMPLE T-TEST
################################################################################

print("\n" + "=" * 50)
print("RESEARCH QUESTION 3")
print("Two-Sample T-Test: Synthetic Opioid Death Rates Over Time")
print("=" * 50)

synthetic_data = overdose_clean[
    (overdose_clean["PANEL"] == "Drug overdose deaths involving other synthetic opioids (other than methadone)") &
    (overdose_clean["ESTIMATE"].notna())
].copy()

synthetic_data["Era"] = synthetic_data["YEAR"].apply(lambda y:
    "Pre-Fentanyl Era (1999-2013)" if 1999 <= y <= 2013 else
    "Fentanyl Era (2014-2018)"     if 2014 <= y <= 2018 else None
)
synthetic_data = synthetic_data[synthetic_data["Era"].notna()]

# Hypothesis test
pre  = synthetic_data[synthetic_data["Era"] == "Pre-Fentanyl Era (1999-2013)"]["ESTIMATE"]
post = synthetic_data[synthetic_data["Era"] == "Fentanyl Era (2014-2018)"]["ESTIMATE"]

t_stat3, p_value3 = stats.ttest_ind(post, pre, equal_var=False)

print(f"\nH0: μ_pre = μ_fentanyl | H1: μ_fentanyl > μ_pre | α = 0.05")
print(f"t-statistic : {t_stat3:.3f}")
print(f"p-value     : {p_value3:.4e}")
print(f"Decision    : {'REJECT H0' if p_value3 < 0.05 else 'FAIL TO REJECT H0'}")

# Visualization - Time Series
ts_synth = synthetic_data.groupby("YEAR")["ESTIMATE"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts_synth["YEAR"], ts_synth["ESTIMATE"], color="#3182bd", linewidth=1.5, marker="o")
ax.set_title("Synthetic Opioid Death Rates: The Fentanyl Crisis", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Death Rate per 100,000")
plt.tight_layout()
plt.savefig("Q3_Synthetic_TimeSeries.png", dpi=300)
plt.show()

print("\n" + "=" * 50)
print("SUMMARY TABLE")
print("=" * 50)
summary = pd.DataFrame({
    "Question": ["Q1", "Q2", "Q3"],
    "t-stat": [t_stat, t_stat2, t_stat3],
    "p-value": [p_value, p_value2, p_value3]
})
print(summary)
print("\n========== END OF ANALYSIS ==========")