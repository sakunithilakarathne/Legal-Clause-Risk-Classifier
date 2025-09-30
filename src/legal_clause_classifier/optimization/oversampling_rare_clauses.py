import os
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset, concatenate_datasets
from config import (ARTIFACTS_DIR, TRAIN_PARQUET_PATH, Y_TRAIN_PATH)



def load_parquet_dataset(parquet_path, y_path):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    y_array = np.load(y_path, allow_pickle=True).astype("float32")
    ds = Dataset.from_pandas(df)
    # Add labels column
    ds = ds.add_column("labels", list(y_array))
    from datasets import Value, Sequence
    ds = ds.cast_column("labels", Sequence(Value("float32")))
    return ds, y_array

def oversample_multi_label_dataset():

    min_count=150
    dataset, y_array = load_parquet_dataset(TRAIN_PARQUET_PATH, Y_TRAIN_PATH)
    
    n_labels = y_array.shape[1]
    counts = y_array.sum(axis=0)  # total number of samples per label
    print("Before oversampling per class:", counts)

    datasets_to_concat = [dataset]
    labels_to_concat = [y_array]

    for label_idx in range(n_labels):
        count = counts[label_idx]
        if count < min_count:
            n_to_add = int(min_count - count)
            # Find all indices where this label is present
            label_indices = np.where(y_array[:, label_idx] == 1)[0]
            # Randomly sample (with replacement)
            extra_indices = np.random.choice(label_indices, size=n_to_add, replace=True)
            # Append duplicates
            datasets_to_concat.append(dataset.select(extra_indices.tolist()))
            labels_to_concat.append(y_array[extra_indices])

    # Merge datasets
    oversampled_dataset = concatenate_datasets(datasets_to_concat)
    oversampled_labels = np.vstack(labels_to_concat)
    print("After oversampling per class:", oversampled_labels.sum(axis=0))

    # Replace the existing 'labels' column instead of adding
    oversampled_dataset = oversampled_dataset.remove_columns("labels")
    oversampled_dataset = oversampled_dataset.add_column("labels", list(oversampled_labels))
    
    oversampled_dataset.save_to_disk(os.path.join(ARTIFACTS_DIR, "train_oversampled"))
    np.save(os.path.join(ARTIFACTS_DIR, "y_train_oversampled.npy"), oversampled_labels)

    print("Oversampled datasets saved!")

