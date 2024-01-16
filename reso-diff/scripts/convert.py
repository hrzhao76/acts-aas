# this script converts the Spacepoint (x, y, z) to (r, phi, z)

import argparse
from pathlib import Path 
import numpy as np 

csv_path = Path("/workspace/acts-aas/reso-diff/run/test_npu0_smear/train_all/event000000000-spacepoint.csv")
output_csv_path = csv_path.parent / (csv_path.stem + "-converted" + csv_path.suffix) 
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

r = np.sqrt(np.power(data[:,2],2) + np.power(data[:,3],2)) / 1000 
phi = np.arctan2(data[:,3], data[:,2]) / 3.14 
z = data[:,4] / 1000 

transformed_data = np.column_stack((r, phi, z))
np.savetxt(output_csv_path, transformed_data, delimiter=',')
