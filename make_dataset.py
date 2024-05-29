import numpy as np
import pandas as pd
import os

SAMPLES = 1000
TIMESTAMPS = 500
DIR = "datasets"

# making the complete dataset
og_data = []
for _ in range(SAMPLES):
    df = pd.DataFrame()

    df["x1"] = np.ones(TIMESTAMPS, dtype=int) * 2
    df["x2"] = np.random.randint(0, 10000, size=TIMESTAMPS) / 10000
    df["x3"] = 2 + np.random.randint(0, 10000, size=TIMESTAMPS) / 10000
    df["x4"] = 4 + np.random.randint(0, 10000, size=TIMESTAMPS) / 5000

    og_data.append(df.values)

og_data = np.array(og_data)

indices = np.arange(len(og_data))
np.random.shuffle(indices)
train_size = int(0.8 * len(og_data))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

missing_data = [0, 0.2, 0.4, 0.6, 0.8]
for p in missing_data:
    data = np.copy(og_data)
    path = f"{DIR}/{int(p*100)}"
    os.makedirs(path, exist_ok=True)
    
    # Generate missing data
    for i in range(data.shape[0]):
        for c in range(data.shape[-1]):
            num_missing = int(p * data.shape[1])  # Number of missing time-points to generate
            missing_indices = np.random.choice(data.shape[1], size=num_missing, replace=False)  # Randomly select indices
            data[i,missing_indices,c] = np.nan  # Set selected indices to NaN
    
    # Save train, validation, and test sets
    train = data[train_indices,:,:]
    test = data[test_indices,:,:]

    print(train.shape)
    print(test.shape)

    np.save(os.path.join(path, "train.npy"), train)
    np.save(os.path.join(path, "test.npy"), test)