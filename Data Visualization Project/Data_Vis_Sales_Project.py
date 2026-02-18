import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

np.random.seed(42)

# TASK 1: EXPLORE THE POPULATION DISTRIBUTION

df = pd.read_csv("E265_CP_file.csv")

column_name = df.columns[0]

plt.figure(figsize=(10, 6))
plt.hist(df[column_name], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Amount Spent (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Population Distribution of Customer Spending', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('population_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("TASK 1: POPULATION DISTRIBUTION ANALYSIS")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Column name: {column_name}")
print(f"\nBasic statistics:")
print(df[column_name].describe())

mean_val = df[column_name].mean()
median_val = df[column_name].median()
print(f"\nMean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")

if mean_val > median_val:
    print("\nDistribution Analysis: RIGHT-SKEWED (positively skewed)")
    print("The mean is greater than the median, indicating a longer tail on the right side.")
elif mean_val < median_val:
    print("\nDistribution Analysis: LEFT-SKEWED (negatively skewed)")
    print("The mean is less than the median, indicating a longer tail on the left side.")
else:
    print("\nDistribution Analysis: SYMMETRIC")
    print("The mean equals the median.")

print("\nNormality Assessment:")
print("Based on the histogram, the distribution does NOT appear to be normally distributed.")
print("A normal distribution would be bell-shaped and symmetric.")

# TASK 2: DRAW A RANDOM SAMPLE

sample = np.random.choice(df[column_name], size=100, replace=False)

mean_sample = np.mean(sample)
sd_sample = np.std(sample, ddof=1)

print("\n" + "=" * 60)
print("TASK 2: RANDOM SAMPLE ANALYSIS")
print("=" * 60)
print(f"Sample size: 100")
print(f"Sample mean: ${mean_sample:.2f}")
print(f"Sample standard deviation: ${sd_sample:.2f}")

# TASK 3: CONSTRUCT A 95% CONFIDENCE INTERVAL

confidence_level = 0.95
degrees_freedom = len(sample) - 1
sample_mean = np.mean(sample)
standard_error = sd_sample / np.sqrt(len(sample))

ci_low_95, ci_high_95 = t.interval(
    confidence_level, 
    df=degrees_freedom, 
    loc=sample_mean, 
    scale=standard_error
)

print("\n" + "=" * 60)
print("TASK 3: 95% CONFIDENCE INTERVAL")
print("=" * 60)
print(f"95% Confidence Interval: (${ci_low_95:.2f}, ${ci_high_95:.2f})")
print(f"Margin of Error: ${(ci_high_95 - ci_low_95) / 2:.2f}")
print(f"Interpretation: We are 95% confident that the true population mean")
print(f"falls between ${ci_low_95:.2f} and ${ci_high_95:.2f}")

# TASK 4: SAMPLE SIZE DETERMINATION

confidence_level_99 = 0.99
margin_error_desired = 5
z_critical = t.ppf((1 + confidence_level_99) / 2, df=degrees_freedom)

required_sample_size = ((z_critical * sd_sample) / margin_error_desired) ** 2
required_sample_size = int(np.ceil(required_sample_size))

print("\n" + "=" * 60)
print("TASK 4: SAMPLE SIZE DETERMINATION")
print("=" * 60)
print(f"Desired margin of error: $5")
print(f"Desired confidence level: 99%")
print(f"Estimated standard deviation (from sample): ${sd_sample:.2f}")
print(f"Required sample size: {required_sample_size}")

if required_sample_size > len(df):
    required_sample_size = len(df)
    print(f"Note: Required sample size exceeds population size.")
    print(f"Using entire population: {required_sample_size}")

new_sample = np.random.choice(df[column_name], size=required_sample_size, replace=False)
new_sample_mean = np.mean(new_sample)
new_sample_sd = np.std(new_sample, ddof=1)
new_standard_error = new_sample_sd / np.sqrt(len(new_sample))
new_degrees_freedom = len(new_sample) - 1

ci_low_99, ci_high_99 = t.interval(
    confidence_level_99,
    df=new_degrees_freedom,
    loc=new_sample_mean,
    scale=new_standard_error
)

print(f"\nNew sample statistics:")
print(f"Sample size: {len(new_sample)}")
print(f"Sample mean: ${new_sample_mean:.2f}")
print(f"Sample standard deviation: ${new_sample_sd:.2f}")
print(f"\n99% Confidence Interval: (${ci_low_99:.2f}, ${ci_high_99:.2f})")
print(f"Margin of Error: ${(ci_high_99 - ci_low_99) / 2:.2f}")

# TASK 5: TRUE POPULATION PARAMETERS

mean_pop = df[column_name].mean()
sd_pop = df[column_name].std(ddof=0)

print("\n" + "=" * 60)
print("TASK 5: TRUE POPULATION PARAMETERS")
print("=" * 60)
print(f"Population size: {len(df)}")
print(f"True population mean: ${mean_pop:.2f}")
print(f"True population standard deviation: ${sd_pop:.2f}")

# TASK 6: CONCLUSIONS AND COMPARISON

print("\n" + "=" * 60)
print("TASK 6: COMPARISON AND CONCLUSIONS")
print("=" * 60)
print("\nCOMPARISON TABLE:")
print("-" * 60)
print(f"{'Metric':<40} {'Value':<20}")
print("-" * 60)
print(f"{'True Population Mean':<40} ${mean_pop:.2f}")
print(f"{'Sample Mean (n=100)':<40} ${mean_sample:.2f}")
print(f"{'New Sample Mean (n=' + str(required_sample_size) + ')':<40} ${new_sample_mean:.2f}")
print("-" * 60)
print(f"{'True Population Std Dev':<40} ${sd_pop:.2f}")
print(f"{'Sample Std Dev (n=100)':<40} ${sd_sample:.2f}")
print(f"{'New Sample Std Dev (n=' + str(required_sample_size) + ')':<40} ${new_sample_sd:.2f}")
print("-" * 60)
print(f"{'95% CI (n=100)':<40} (${ci_low_95:.2f}, ${ci_high_95:.2f})")
print(f"{'99% CI (n=' + str(required_sample_size) + ')':<40} (${ci_low_99:.2f}, ${ci_high_99:.2f})")
print("-" * 60)
print(f"{'Required Sample Size for 99% CI':<40} {required_sample_size}")
print(f"{'with Margin of Error = $5':<40}")
print("-" * 60)

error_100 = abs(mean_sample - mean_pop)
error_new = abs(new_sample_mean - mean_pop)

print(f"\nAccuracy Analysis:")
print(f"Error in sample mean (n=100): ${error_100:.2f}")
print(f"Error in new sample mean (n={required_sample_size}): ${error_new:.2f}")

contains_mean_95 = ci_low_95 <= mean_pop <= ci_high_95
contains_mean_99 = ci_low_99 <= mean_pop <= ci_high_99

print(f"\nConfidence Interval Coverage:")
print(f"95% CI contains true mean: {contains_mean_95}")
print(f"99% CI contains true mean: {contains_mean_99}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)