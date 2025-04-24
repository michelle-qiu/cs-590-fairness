#from functional_dependency import parse, load_fdset
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import sys
sys.path.append('../src')
from compute_error import display_violation, write_violation
from write_rc_distribution import write_dist
import os


def get_column_range(column_ls, original):
    column_range = {}
    for col in column_ls:
        column_range[col] = set(original[col].unique())
    return column_range

def add_error(original, size, err_num, column_ls,column_range, errType = '1', seed=42):
    #used for adding error randomly to specific groups, return df and err_tracker
    #err_rate means the # of error cells, not # of error tuples
    np.random.seed(seed)
    random.seed(seed)
    clean = original.sample(size)
    dirty = clean.copy()
    # for clean, creat a new column to indicate the dirty value
    for col in column_ls:
        clean[col + '_dirty'] = None

    err_tracker = set()
    i = 0
    while i < err_num: #the number of dirty cells
        row_idx = np.random.choice(dirty.index)
        ID = dirty.loc[row_idx, 'ID']
        col = np.random.choice(column_ls)

        if (row_idx,col) not in err_tracker: #if we haven't already introduced error for this particular row and column
            if errType == '1': 
                new_col = np.random.choice(list(column_range[col] - set([dirty[col][row_idx]])))
            elif errType == '2':
                new_col = 7
            dirty[col][row_idx] = new_col 
            clean[col+'_dirty'][row_idx]=new_col
            err_tracker.add((ID,col))
            i += 1
            #print(f"row_idx: {row_idx}, col: {col}")
            #print(f"fixed[col][row_idx]: {dirty[col][row_idx]}")
    #compute the length of err_tracker and the unique row_idx in err_tracker
    print('number of error cells:', len(err_tracker))
    print('number of error tuples:', len(set([x[0] for x in err_tracker])))
    return clean, dirty, err_tracker

def add_error_random(clean, err_num, column_ls,column_range, errType = '1', seed=42):
    #used for adding error randomly to specific groups, return df and err_tracker
    #err_rate means the # of error cells, not # of error tuples
    np.random.seed(seed)
    random.seed(seed)
    #clean = original.sample(size)
    dirty = clean.copy()
    # for clean, creat a new column to indicate the dirty value
    for col in column_ls:
        clean[col + '_dirty'] = None

    err_tracker = set()
    i = 0
    while i < err_num:
        row_idx = np.random.choice(dirty.index)
        ID = dirty.loc[row_idx, 'ID']
        col = np.random.choice(column_ls)

        if (row_idx,col) not in err_tracker:
            if errType == '1':
                new_col = np.random.choice(list(column_range[col] - set([dirty[col][row_idx]])))
            elif errType == '2':
                new_col = 7
            dirty[col][row_idx] = new_col
            clean[col+'_dirty'][row_idx]=new_col
            err_tracker.add((ID,col))
            i += 1
            #print(f"row_idx: {row_idx}, col: {col}")
            #print(f"fixed[col][row_idx]: {dirty[col][row_idx]}")
    #compute the length of err_tracker and the unique row_idx in err_tracker
    print('number of error cells:', len(err_tracker))
    print('number of error tuples:', len(set([x[0] for x in err_tracker])))
    return clean, dirty, err_tracker

if __name__ == '__main__':
    #print('current path:', os.getcwd())
    #original = pd.read_csv('pums_info/original_idx_large.csv')
    #sample from a larger source dataset
    original = pd.read_csv('pums_info/original_idx_large3.csv')
    #print(len(original))
    size_ls = [4000,6000,8000,10000]
    
    err_rate_ls = [0.05,0.1]
    #ST, DIV, CIT, RAC1P, RAC2P, POBP, and WAOB are in short for state code, division, citizenship status, race code 1, race code 2, place of birth, world area of birth respectively for dataset ACS
    column_ls = ['REGION', 'ST', 'CIT', 'NATIVITY','RAC1P','RAC2P','POBP','WAOB'] #2 options: only RHS or both LHS and RHS
    column_range = get_column_range(column_ls, original)
    bygroup = False #add error to different groups separately or not 
    rc = 'NATIVITY'  
    distribution_col_ls = ['NATIVITY'] #columns that we want to record the distribution
    err_ver = '1'
    #seed = 42
    fd_ls = ['newFDs'] #functional dependency list 
    err_pattern = '2080'
    for seed in [42,66]:
        if seed == 42:
            folder = '1_seed42'
        elif seed == 66:
            folder = '2_seed66'

        if bygroup == True:
            base_dir = f'datagen_new{folder}/pums_bygroup5050_error{err_pattern}_{rc.lower()}_ver{err_ver}_newFDs/'
        else:
            base_dir = f'datagen_new{folder}/pums_random_error{err_pattern}_{rc.lower()}_ver{err_ver}_newFDs/'
        #print('base_dir:', base_dir)
        if not os.path.exists(base_dir):
            # make sure the directory path exists
            os.makedirs(base_dir)

        for size in size_ls:
            for err_rate in err_rate_ls:
                if bygroup == True: #if we want to introduce error by group
                    dirPath = base_dir+f"{size}_{err_rate}_"
                    err_num = int(size * err_rate * len(column_ls))
                    ori_dis = original[original[rc] ==2]  #here rc = nativity
                    ori_nodis = original[original[rc]==1]
                    dis_dict = {'size': int(size*0.5),
                                'err_num': int(err_num*0.8)}
                    nodis_dict = {'size':size - dis_dict['size'], #size 
                                'err_num':err_num-dis_dict['err_num']}
                    clean_dis, dirty_dis, err_tracker_dis = add_error(ori_dis, dis_dict['size'], dis_dict['err_num'],column_ls, column_range, errType = err_ver, seed = seed)
                    clean_nodis, dirty_nodis, err_tracker_nodis = add_error(ori_nodis, nodis_dict['size'], nodis_dict['err_num'], column_ls, column_range, errType = err_ver, seed = seed)

                    #combine
                    clean = pd.concat([clean_dis, clean_nodis], ignore_index=True)
                    dirty = pd.concat([dirty_dis, dirty_nodis], ignore_index=True)
                    err_tracker = err_tracker_dis.union(err_tracker_nodis)

                else: # add error randomly to all groups
                    dirPath = base_dir + f"{size}_{err_rate}_"
                    err_num = int(size * err_rate * len(column_ls)) #eg 2000*0.05*8?
                    print(f"error num is {err_num}")
                    print(f"size is {size}, err_rate is {err_rate}, column_ls is {column_ls}")
                    np.random.seed(seed)
                    random.seed(seed)
                    clean = original.sample(size)

                    ori_dis = clean[clean[rc] ==2]
                    ori_nodis = clean[clean[rc]==1]
                    dis_err_num = int(err_num*0.3529) #define the noise distribution here
                    nodis_err_num = err_num - dis_err_num
                    clean_dis, dirty_dis, err_tracker_dis = add_error_random(ori_dis, dis_err_num,column_ls, column_range, errType = err_ver, seed = seed)
                    clean_nodis, dirty_nodis, err_tracker_nodis = add_error_random(ori_nodis, nodis_err_num, column_ls, column_range, errType = err_ver, seed = seed)

                    #combine
                    clean = pd.concat([clean_dis, clean_nodis], ignore_index=True)
                    dirty = pd.concat([dirty_dis, dirty_nodis], ignore_index=True)
                    err_tracker = err_tracker_dis.union(err_tracker_nodis)
                    
                    
                clean.to_csv(dirPath + f'clean.csv', index=False)
                dirty.to_csv(dirPath + f'dirty.csv', index=False)
                with open(dirPath + f'err_tracker.txt', 'w') as file:
                    file.write(f'number of error cells: {len(err_tracker)}\n')
                    file.write(f'number of error tuples: {len(set([x[0] for x in err_tracker]))}\n')
                    file.write(f'rate of error cells: {len(err_tracker) / (size * len(column_ls))}\n')
                    file.write(f'rate of error tuples: {len(set([x[0] for x in err_tracker])) / size}\n')
                    file.write(str(err_tracker))

                
                #write distribution of rc columns
                suffix = 'original'
                for rc_col in distribution_col_ls:
                    rc_dist = clean[rc_col].value_counts(normalize=True)*100
                    rc_dist = rc_dist.sort_index()
                    write_dist(dirPath, rc_col, rc_dist, suffix = suffix)

                #write violation ratios of different FDs
                write_violation(dirPath, dirty,clean,fd_ls)
                for fd in fd_ls:
                    display_violation(dirPath, dirty, clean, fd_ls, column_ls)
                seed = seed + 1

'''
fd_filename = 'datagen/pums_data/syn_pums_data_new/FDs.txt'
delta = load_fds(fd_filename)
column_ls = set()
for fd in delta.fds:
    for col in fd.lhs.cols:
        column_ls.add(col)

    column_ls.add(fd.rhs.col)


size_ls = [1000,2000]
err_ls = [0.1, 0.4]
for size, err in enumerate((size_ls,err_ls)):
    print(size, err)
import pandas as pd
def make_col_int(df, *args):
    for col in args:
        df[col] = df[col].astype(float)
        df[col] = df[col].astype(int)
def make_col_string(df, *args):
    for col in args:
        df[col] = df[col].astype(str)
df = pd.read_csv('synthetic_data_version_1_new_1000.csv')
make_col_int(df, 'RAC1P', 'SEX', 'REGION', 'ST', 'CIT', 'NATIVITY', 'DIS')
make_col_string(df, 'PINCP', 'COW', 'MSP', 'SCHL', 'MIL')
df.to_csv('synthetic_data_version_1_new_1000.csv', index=False)
'''