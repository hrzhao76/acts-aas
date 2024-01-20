# this script converts the Spacepoint (x, y, z) to (r, phi, z)

import argparse
from pathlib import Path

import numpy as np


def process_csv_and_convert(csv_path: Path, output_csv_path: Path = None):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    transformed_data = convert_coordinates(data)

    if output_csv_path is None:
        output_csv_path = csv_path.parent / (
            csv_path.stem + "-converted" + csv_path.suffix
        )

    np.savetxt(output_csv_path, transformed_data, delimiter=",")


def convert_coordinates(original_data):
    r = (
        np.sqrt(np.power(original_data[:, 2], 2) + np.power(original_data[:, 3], 2))
        / 1000
    )
    phi = np.arctan2(original_data[:, 3], original_data[:, 2]) / 3.14
    z = original_data[:, 4] / 1000

    transformed_data = np.column_stack((r, phi, z))
    return transformed_data


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
