import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the text files
directory = "./"  # Use the current directory
files = sorted(
    [f for f in os.listdir(directory) if f.endswith(".txt")]
)  # Get all .txt files

# Parameter names and their default values
param_names = ["Tau", "ASocIso", "BSocIso", "Lambda", "ASocMean", "BSocMean", "VD"]
default_values = [0.4, 2.72, 0.2, 0.176, 0.4, 2.8, 3]

# Initialize a list to store parameter data from each file
param_data = {
    file: [[] for _ in range(7)] for file in files
}  # Dictionary to store data per file
best_solution_data = {
    file: [] for file in files
}  # Dictionary to store best solution values per file

# Define distinct colors for different files
colors = sns.color_palette("tab10", len(files))

# Define different line styles for visual distinction
line_styles = ["-", "--", "-.", ":", "-"]

# Read and extract data from each file
for file_name in files:
    file_path = os.path.join(directory, file_name)

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Simulation:"):
                match = re.search(r"\[(.*?)\]", line)  # Extract values inside brackets
                if match:
                    values = match.group(1).split(", ")
                    for i in range(7):  # Store each parameter separately
                        param_data[file_name][i].append(float(values[i].strip("'")))

            # Extract the best solution values from the line containing "Parameters of the best solution"
            elif "Parameters of the best solution" in line:
                match_best = re.search(r"\[(.*?)\]", line)
                if match_best:
                    best_values = match_best.group(1).split(", ")
                    best_solution_data[file_name] = [
                        float(val.strip("'")) for val in best_values
                    ]

# Plot KDE for each parameter
plt.figure(figsize=(15, 10))
for param_index in range(7):
    plt.subplot(4, 2, param_index + 1)

    for file_index, file_name in enumerate(files):
        # Get the best solution value for the current file
        best_values_legend = f"{best_solution_data[file_name][param_index]:.2f}"

        # Plot KDE line for each file with thicker lines and different line styles
        sns.kdeplot(
            param_data[file_name][param_index],
            label=f"Calib {file_index+1}: {best_values_legend}",
            linestyle=line_styles[file_index % len(line_styles)],
            color=colors[file_index],
            lw=2,  # Set line width to 2 for more prominent lines
        )

    # Mark the default value with a hollow circle
    plt.scatter(
        default_values[param_index],
        0,
        color="black",
        marker="o",
        label=f"Default: {default_values[param_index]}",
    )

    plt.title(f"{param_names[param_index]}")
    plt.ylabel("Density")
    plt.legend()

plt.tight_layout()
plt.show()
