import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from helper import *


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def compare_distributions_from_files(file1, file2):
    """
    Compare two datasets from two files using the Kolmogorov-Smirnov (K-S) test
    and plot KDEs on the same canvas with different line styles.

    Parameters:
    - file1: The first file (CSV).
    - file2: The second file (CSV).

    Returns:
    - ks_statistic: The K-S test statistic (maximum difference between ECDFs).
    - p_value: The p-value for the K-S test.
    - conclusion: Interpretation of the result (whether the distributions are similar).
    """
    try:
        data1 = read_frequency_table(file1)
    except KeyError:
        data1 = read_speed_data(file1)

    try:
        data2 = read_frequency_table(file2)
    except KeyError:
        data2 = read_speed_data(file2)

    print(np.mean(data1), np.mean(data2))
    output = rmspe(data1, data2)
    print(output)

    # Plot KDEs of both datasets on the same canvas with different line styles
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data1, label="actual", fill=True, linestyle="-")
    sns.kdeplot(data2, label="simulated", fill=True, linestyle="--")

    plt.title("KDE Comparison")
    plt.xlabel("Speed")
    plt.ylabel("Density")

    # Place legend in the top-right corner
    plt.legend(loc="upper right")

    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(data1, data2)

    # Conclusion based on p-value
    if p_value > 0.05:
        conclusion = "The distributions are similar (fail to reject H0)."
    else:
        conclusion = "The distributions are different (reject H0)."

    # Display K-S statistic and p-value on the plot (top-left corner)
    plt.text(
        0.95,
        0.30,
        f"RMSPE: {output:.4f}",
        # "RMSPE: {output:.4f}\nK-S Statistic: {ks_statistic:.4f}\np-value: {p_value:.4f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=12,
        color="black",
    )

    plt.show()

    return ks_statistic, p_value, conclusion


compare_distributions_from_files(
    "civil_actual.csv", "pedestrian_speed_civil-subhourly_002.csv"
)
