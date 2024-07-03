import pandas as pd

# map = {'Cubic': 16684, 'Hexagonal': 8898, 'Monoclinic': 27355, 'Orthorhombic': 25065, 'Tetragonal': 14032, 'Triclinic': 14282, 'Trigonal': 10509}
#
# # 读取 CSV 文件为 DataFrame
# df = pd.read_csv('kind_res.csv')
#
# # 遍历每一行
# for index, row in df.iterrows():
#     # 如果 value 大于 300，就置为 300
#     if row['value'] > 300:
#         df.at[index, 'value'] = 300
#     # df.at[index, 'group'] = row['group'] + "\n" + str(map[row['group']])
#
# df.to_csv("kind_res_img.csv", index=False)

# # 读取 CSV 文件为 DataFrame
# df = pd.read_csv('kind_res.csv')
#
# group = df.groupby("group")
#
# map = {}
# for g, v in group:
#     map[g] = v["value"].sum()
# print(map)


map = {'Cubic': 16684, 'Hexagonal': 8898, 'Monoclinic': 27355, 'Orthorhombic': 25065, 'Tetragonal': 14032, 'Triclinic': 14282, 'Trigonal': 10509}


for k, v in map.items():
    print("{}\n{}".format(k, v))