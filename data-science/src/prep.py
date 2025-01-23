# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()

    return args

def main(args):
    '''Read, split, and save datasets'''

    # # Reading Data
    # data = pd.read_csv((Path(args.raw_data)))
    # data = data[FEATURE_COLS + [TARGET_COL]]

    # # Split Data into train, val and test datasets
    # random_data = np.random.rand(len(data))

    # msk_train = random_data < 0.7
    # msk_val = (random_data >= 0.7) & (random_data < 0.85)
    # msk_test = random_data >= 0.85

    # train = data[msk_train]
    # val = data[msk_val]
    # test = data[msk_test]

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False

    mlflow.log_metric('train size', train_df.shape[0])
    mlflow.log_metric('test size', test_df.shape[0])
    mlflow.log_metric('val size', val.shape[0])


     # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False

    # train.to_parquet((Path(args.train_data) / "train.parquet"))
    # val.to_parquet((Path(args.val_data) / "val.parquet"))
    # test.to_parquet((Path(args.test_data) / "test.parquet"))

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()
