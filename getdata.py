import pandas as pd
import numpy as np


def get_data(SimCounter, filename):
    # Construct file names
    input_file = f"{filename}_{SimCounter:03}.pp"
    output_csv = f"gotdata_{filename}_{SimCounter:03}.csv"
    analysis_csv = f"pedestrian_speed_{filename}_{SimCounter:03}.csv"

    # Read the file and extract relevant data
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Find the header line starting with $PEDESTRIAN
    for i, line in enumerate(lines):
        if line.startswith("$PEDESTRIAN"):
            headers = line.strip().replace("$PEDESTRIAN:", "").split(";")
            data_lines = lines[i + 1 :]
            break
    else:
        raise ValueError("$PEDESTRIAN section not found in file")

    # Convert data to DataFrame
    data = [line.strip().split(";") for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=headers)

    # Convert numeric columns to appropriate types
    df["NO"] = df["NO"].astype(int)
    df["SPEED"] = df["SPEED"].astype(float)
    # df["SIMSEC"] = df["SIMSEC"].astype(float)
    # df["DESSPEED"] = df["DESSPEED"].astype(float)
    df["CONSTRELNO"] = df["CONSTRELNO"].astype(int)

    # Save extracted data
    df.to_csv(output_csv, index=False)

    # Filtering for analysis
    df_filtered = df[
        (df["SPEED"] >= 0.2)
        & (df["CONSTRELTYPE"] == "Pedestrian link")
        & df["CONSTRELNO"].isin([1, 2])
    ]

    # Compute average speed and desired speed per pedestrian
    # df_avg = df_filtered.groupby("NO")[["SPEED", "DESSPEED"]].mean().reset_index()
    df_avg = df_filtered.groupby("NO")[["SPEED"]].mean().reset_index()

    # Save analysis results
    df_avg.to_csv(analysis_csv, index=False)

    # Checking directional counts
    df["PEDROUTSTA\\NO"] = df["PEDROUTSTA\\NO"].astype(int)
    flow_counts = df.groupby("PEDROUTSTA\\NO")["NO"].nunique()
    with open("flow_check.txt", "a") as f:
        f.write(
            f"{filename}_{SimCounter:03}, {flow_counts.get(1, 0)}, {flow_counts.get(2, 0)}\n"
        )

    # Return sorted list of average speeds
    return np.average(sorted(df_avg["SPEED"].tolist()))


print(get_data(2, "civil-subhourly"))
