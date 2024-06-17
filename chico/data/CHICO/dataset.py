import os
import pickle
import numpy as np

# Define the directory path
directory = "./dataset/"

# Initialize an empty dictionary to store file contents
data_dict = {}

# Loop through each folder from S00 to S19
for value in range(20):
    folder = f"S{value:02d}"
    folder_path = os.path.join(directory, folder)

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".pkl"):
                # Load pickle file
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                # Add data to dictionary with file name as key
                data_dict[file_name] = data

# Convert dictionary values to object dtype to handle ragged nested sequences
for key, value in data_dict.items():
    if isinstance(value, list) or isinstance(value, tuple):
        data_dict[key] = np.array(value, dtype=object)
print(data_dict)
# Save the dictionary as npz file
np.savez("data_npz_file.npz", **data_dict)
