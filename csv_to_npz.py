import pandas as pd
import numpy as np

file_paths = ["/home/alexis/_svdd/df_train_clean.csv", "/home/alexis/_svdd/df_test_clean.csv", "/home/alexis/_svdd/df_test_snr15.csv", "/home/alexis/_svdd/df_test_snr20.csv", "/home/alexis/_svdd/df_test_snr25.csv"]

for file_path in file_paths:
    df = pd.read_csv(file_path)

    ids = df['filename'].values
    features = df.drop(['filename', 'len_in_sec'], axis=1).values

    npz_file_path = file_path.replace('.csv', '.npz')

    np.savez(npz_file_path, ids=ids, features=features)
    print(f"Saved {npz_file_path}.")