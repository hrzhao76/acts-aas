####################################################
# This script converts the Spacepoint features in  #
# (r, phi, z) coordinates to json for perf_analyzer#
# Author: Haoran Zhao                              #
# Email: haoran.zhao [at] cern.ch                  #
# Date: Jan 2024                                   #
####################################################
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def process_csv_and_convert(csv_path: Path, output_csv_path: Path = None):
    if output_csv_path is None:
        if csv_path.suffix == ".csv":
            output_csv_path = csv_path.parent / f"{csv_path.stem}.json"
        elif csv_path.is_dir():
            output_csv_path = csv_path / f"{csv_path.stem}.json"

    csv_content_list = []
    # Read the CSV file into a pandas DataFrame
    if csv_path.suffix == ".csv" and csv_path.is_file():
        df = pd.read_csv(csv_path)
        assert df.shape[1] == 3, "The input CSV file must have 3 features"
        csv_content_list.append(df)
    elif csv_path.is_dir():
        print(f"Reading all CSV files in {csv_path}")
        for file in csv_path.glob("*-converted.csv"):
            df = pd.read_csv(file)
            csv_content_list.append(df)
    else:
        raise NotImplementedError(f"Unsupported file type: {csv_path.suffix}")

    # Convert the DataFrame to a flattened list
    json_content = convert_json(csv_content_list)
    with open(output_csv_path, "w") as json_file:
        json.dump(json_content, json_file, indent=4)


def convert_json(csv_df_list: list):
    json_format_list = []
    for df in csv_df_list:
        flattened_list = df.values.flatten().tolist()
        json_format_list.append(
            {"FEATURES": {"content": flattened_list, "shape": list(df.shape)}}
        )

    return {"data": json_format_list}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=False, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_csv_path = (
        None if args.output_csv_path is None else Path(args.output_csv_path)
    )
    process_csv_and_convert(csv_path, output_csv_path)
