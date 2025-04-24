#given clean and dirty dataset, write both the overall distribution of rc column & the distribution of rc column in error cells
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import sys
sys.path.append('../src')
from compute_error import display_violation, write_violation
import os

def write_dist(dir, column, distribution, suffix = 'original'):
    filename = dir + f"{column}_RC_{suffix}.txt"
    with open(filename, 'w') as f:
        f.write(f"{column}\n")
        keys_string = ":".join(map(str, distribution.index))
        values_string = ":".join(f"{val:.1f}" for val in distribution) #round to 1 decimal place
        f.write(f"{keys_string}\n")
        f.write(values_string)

def print_dist(column, df):
    distribution = df[column].value_counts(normalize=True)*100
    distribution = distribution.sort_index()
    print(f"RC: {column}")
    keys_string = ":".join(map(str, distribution.index))
    values_string = ":".join(f"{val:.1f}" for val in distribution) # round to 1 decimal place
    print(f"Keys: {keys_string}")
    print(f"Values: {values_string}")
    print("\n")


if __name__ == '__main__':
    size_ls = [500, 1000, 2000, 5000, 10000]
    err_rate_ls = [0.01, 0.05, 0.1, 0.2]
    suffix = 'original'
    distribution_col_ls = ['NATIVITY', 'CIT']
    dir = 'pums_ver2/'
    for size in size_ls:
        for err_rate in err_rate_ls:
            file_name = f'{size}_{err_rate}_dirty.csv'
            dirty = pd.read_csv(dir+file_name)
            #write the distibution of dirty or clean csv
            for rc_col in distribution_col_ls:
                # print(rc_col)
                rc_dist = dirty[rc_col].value_counts(normalize=True) * 100
                rc_dist = rc_dist.sort_index()
                write_dist(dir+f'{size}_{err_rate}_', rc_col, rc_dist, suffix = suffix)