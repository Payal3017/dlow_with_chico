import numpy as np
import os

subjects = [
    "S02",  # test
    "S03",  # test
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S01",
    "S16",
    "S17",
    "S18",  # test
    "S19",  # test
]

data_file = os.path.join("chico", "data", "CHICO", "dataset")
data_ss = {}
index = 0
for key in subjects:
    current_path = os.path.join(data_file, key)
    # Load all pickle files in the current path without "CRASH" in the file name
    files = [
        f for f in os.listdir(current_path) if not "CRASH" in f and f.endswith(".pkl")
    ]

    for file in files:
        # Load the data from the pickle file
        data = np.load(os.path.join(current_path, file), allow_pickle=True)

        # Get the base name without the extension
        base_name = os.path.splitext(file)[0]

        data = np.array(data, dtype=object)
        flattened_data = data.reshape(-2)

        if index == 0:
            print(len(flattened_data))
        index += 1
        # Initialize the list in the dictionary if not already present
        if key not in data_ss:
            data_ss[key] = {}

        if base_name not in data_ss[key]:
            data_ss[key][base_name] = []

        # Append the loaded data to the list in the dictionary
        data_ss[key][base_name] = data[0]

# Flatten the lists in the data_ss dictionary
# for key in data_ss:
#     data_ss[key] = [item for sublist in data_ss[key] for item in sublist]

# Handle inhomogeneous arrays by converting to object arrays
data_np = {key: value for key, value in data_ss.items()}
data_np = {
    base_name: {
        action: np.array(data_ss[base_name][action], dtype=object)
        for action in data_ss[base_name]
    }
    for base_name in data_ss
}
# Save the data_ss dictionary as a .npz file
np.savez("data_3d_chico.npz", positions_3d=data_np)
