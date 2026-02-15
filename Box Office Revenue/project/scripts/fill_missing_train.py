import pandas as pd

train = pd.read_csv('./data/train.csv')

train.loc[train['id'] == 16, 'revenue'] = 192864
train.loc[train['id'] == 90, 'budget'] = 30000000
train.loc[train['id'] == 118, 'budget'] = 60000000
train.loc[train['id'] == 149, 'budget'] = 18000000
train.loc[train['id'] == 313, 'revenue'] = 12000000
train.loc[train['id'] == 451, 'revenue'] = 12000000
train.loc[train['id'] == 464, 'budget'] = 20000000
train.loc[train['id'] == 470, 'budget'] = 13000000
train.loc[train['id'] == 513, 'budget'] = 1100000
train.loc[train['id'] == 797, 'budget'] = 8000000
train.loc[train['id'] == 819, 'budget'] = 90000000
train.loc[train['id'] == 850, 'budget'] = 1500000
train.loc[train['id'] == 1007, 'budget'] = 2
train.loc[train['id'] == 1112, 'budget'] = 7500000
train.loc[train['id'] == 1131, 'budget'] = 4300000
train.loc[train['id'] == 1359, 'budget'] = 10000000
train.loc[train['id'] == 1542, 'budget'] = 1
train.loc[train['id'] == 1570, 'budget'] = 15800000
train.loc[train['id'] == 1571, 'budget'] = 4000000
train.loc[train['id'] == 1714, 'budget'] = 46000000
train.loc[train['id'] == 1721, 'budget'] = 17500000
train.loc[train['id'] == 1865, 'revenue'] = 25000000
train.loc[train['id'] == 1885, 'budget'] = 12
train.loc[train['id'] == 2091, 'budget'] = 10
train.loc[train['id'] == 2268, 'budget'] = 17500000
train.loc[train['id'] == 2491, 'budget'] = 6
train.loc[train['id'] == 2602, 'budget'] = 31000000
train.loc[train['id'] == 2612, 'budget'] = 15000000
train.loc[train['id'] == 2696, 'budget'] = 10000000
train.loc[train['id'] == 2801, 'budget'] = 10000000
train.loc[train['id'] == 335, 'budget'] = 2
train.loc[train['id'] == 348, 'budget'] = 12
train.loc[train['id'] == 640, 'budget'] = 6
train.loc[train['id'] == 696, 'budget'] = 1
train.loc[train['id'] == 1199, 'budget'] = 5
train.loc[train['id'] == 1282, 'budget'] = 9
train.loc[train['id'] == 1347, 'budget'] = 1
train.loc[train['id'] == 1755, 'budget'] = 2
train.loc[train['id'] == 1801, 'budget'] = 5
train.loc[train['id'] == 1918, 'budget'] = 592
train.loc[train['id'] == 2033, 'budget'] = 4
train.loc[train['id'] == 2118, 'budget'] = 344
train.loc[train['id'] == 2252, 'budget'] = 130
train.loc[train['id'] == 2256, 'budget'] = 1

train = train.dropna(subset=['revenue'])

train.to_csv('./data/train.csv', index=False)