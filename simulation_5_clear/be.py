import os, glob

ts_folder = "C:/Users/reawe/Desktop/Vienna/simulation_5_clear/timeseries"
sim_id = 1
pattern = os.path.join(ts_folder, f"sim_{sim_id:04d}_rep_*.csv")

print("Pattern:", pattern)
print("Files found:", glob.glob(pattern))
