import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

crystal_systems = {'triclinic': [0, 1],
                   'monoclinic': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                   'orthorhombic1': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                   'orthorhombic2': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                   'orthorhombic3': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
                   'tetragonal1': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
                   'tetragonal2': [56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
                   'trigonal': [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
                   'hexagonal': [78, 79, 80, 81, 82, 83, 84, 85, 86],
                   'cubic': [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]}

crystal_systems_result = {}

for idx, item in enumerate(crystal_systems.values()):
    for i in item:
        crystal_systems_result[i] = idx



# 读取CSV文件
df = pd.read_csv('matsum.csv')

# 提取第二列到第十二列
subset_df = df.iloc[:, 2:12]
subset_df.columns = range(1, 11)
# print(subset_df)
# 找到最大值索引

def my_function(value):
    return crystal_systems_result[value]

max_indexes = subset_df.astype(float).idxmax(axis=1)
subset_df = df.iloc[:, 0]
subset_df = subset_df.apply(my_function) + 1

df_combined = pd.concat([subset_df, max_indexes], axis=1)
# print(df_combined.head(6000))
df_combined.columns = ["1", "2"]
count_equal = (df_combined["1"] == df_combined["2"]).sum()
# print(count_equal)
# print(df_combined.shape)
# print(df_combined.head(50))