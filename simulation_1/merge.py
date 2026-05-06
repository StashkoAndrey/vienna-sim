import os
import pandas as pd
import re
from collections import defaultdict

# --- Paths ---
root_dir = "C:/Users/reawe/Desktop/Vienna/simulation/batch_results"
output_dir = "C:/Users/reawe/Desktop/Vienna/simulation/merged"
os.makedirs(output_dir, exist_ok=True)

# Regex to extract all needed values from folder name
pattern = r"N(?P<N>[\d.]+)_E(?P<E>[\d.]+)_U(?P<U>[\d.]+)_D(?P<D>[\d.]+)_R(?P<R>\d+)"

# Dictionary to group folders by (N, E, U, D)
groups = defaultdict(list)

# Step 1: Group folders
for folder in os.listdir(root_dir):
    match = re.match(pattern, folder)
    if not match:
        continue
    info = match.groupdict()
    group_key = (info['N'], info['E'], info['U'], info['D'])
    groups[group_key].append((folder, info['R']))

# Step 2: Merge each group
for (n, e, u, d), folders in groups.items():
    merged_frames = []

    for folder, r_val in folders:
        folder_path = os.path.join(root_dir, folder)

        # Find the subfolder
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        if not subfolders:
            continue
        inner_folder = os.path.join(folder_path, subfolders[0])

        # Read and append all CSVs in the subfolder
        for file in os.listdir(inner_folder):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(inner_folder, file))
                    df['replicate'] = int(r_val)
                    df['folder'] = folder
                    df['file'] = file
                    merged_frames.append(df)
                except Exception as e:
                    print(f"Error in {folder}/{file}: {e}")

    if merged_frames:
        batch_df = pd.concat(merged_frames, ignore_index=True)
        out_name = f"merged_N{n}_E{e}_U{u}_D{d}.csv"
        out_path = os.path.join(output_dir, out_name)
        batch_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")
    else:
        print(f"⚠️ No data for batch N{n}_E{e}_U{u}_D{d}")

print("✅ All batches processed.")
