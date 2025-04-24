import pandas as pd

# original = pd.read_csv('pums_info/original_idx_large.csv')
# #set value to 1 if 'REGION' = 1 or 3
# original['REGION'] = original['REGION'].apply(lambda x: 1 if x == 1 or x == 3 else 2)
# print(original.head(20))
# original.to_csv('pums_info/original_idx_large_region2.csv', index=False)
original = pd.read_csv('pums_info/original_idx_large_region2.csv')
print(original['REGION'].unique())
