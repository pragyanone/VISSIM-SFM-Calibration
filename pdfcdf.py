import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_frequency_table(filename):
    """Read CSV file containing speed and frequency data."""
    df = pd.read_csv(filename)
    data = np.repeat(df["speed"].values, df["no"].values)
    return data[data > 0]  # Ensure all values are positive


def read_speed_data(filename):
    """Read CSV file containing NO, SPEED, and DESSPEED data."""
    df = pd.read_csv(filename)
    return df["SPEED"].values  # Extract only the speed column


def compute_pdf_cdf(data, bins=50):
    """Compute and plot PDF & CDF from raw speed data."""
    # Sort data for CDF calculation
    sorted_data = np.sort(data)

    # Compute histogram for PDF
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

    # Compute CDF
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    return bin_centers, counts, sorted_data, cdf_values


def generate_pdf_cdf_table(pdf_x, pdf_y, cdf_x, cdf_y, interval=0.5):
    """Generate a table of PDF & CDF values at fixed speed intervals."""
    min_speed = min(cdf_x)
    max_speed = max(cdf_x)

    # Start from a multiple of 0.25 just less than the minimum speed
    start = np.floor(min_speed / interval) * interval

    # End at the maximum speed
    stop = np.ceil(max_speed / interval) * interval

    speed_intervals = np.arange(start, stop + interval, interval)  # Fixed intervals

    # Interpolate PDF and CDF values at fixed intervals
    pdf_interpolated = np.interp(speed_intervals, pdf_x, pdf_y)
    cdf_interpolated = np.interp(speed_intervals, cdf_x, cdf_y)

    # Round values to 2 decimal places
    pdf_interpolated_rounded = np.round(pdf_interpolated, 2)
    cdf_interpolated_rounded = np.round(cdf_interpolated, 2)

    # Create the table
    table_df = pd.DataFrame(
        {
            "Speed": np.round(speed_intervals, 2),
            "PDF": pdf_interpolated_rounded,
            "CDF": cdf_interpolated_rounded,
        }
    )

    # Print table to console
    print(table_df)

    # Save to CSV
    csv_filename = f"5_pdf_cdf_table_{filename}.csv"
    table_df.to_csv(csv_filename, index=False)
    print(f"\nTable saved as '{csv_filename}'.")

    return table_df


def plot_pdf_cdf(speed_intervals, pdf_values, cdf_values):
    """Plot interpolated PDF and CDF on the same graph."""
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot PDF as a bar chart
    ax1.bar(
        speed_intervals, pdf_values, width=0.2, alpha=0.5, color="blue", label="PDF"
    )
    ax1.set_xlabel("Speed")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper left")

    # Plot CDF on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        speed_intervals,
        cdf_values,
        color="green",
        linestyle="--",
        marker="o",
        label="CDF",
    )
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend(loc="upper right")

    plt.title("Interpolated PDF & CDF")
    plt.grid()
    plt.show()


# Example usage:
filename = "valid_civil_actual.csv"  # Change as needed

try:
    speed_data = read_frequency_table(filename)
except KeyError:
    speed_data = read_speed_data(filename)

# Compute PDF & CDF
pdf_x, pdf_y, cdf_x, cdf_y = compute_pdf_cdf(speed_data)

# Generate PDF & CDF table
pdf_cdf_table = generate_pdf_cdf_table(pdf_x, pdf_y, cdf_x, cdf_y)

# Plot the final interpolated PDF & CDF
plot_pdf_cdf(pdf_cdf_table["Speed"], pdf_cdf_table["PDF"], pdf_cdf_table["CDF"])
