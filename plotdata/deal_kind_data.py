import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv('data.csv')
new_df = pd.DataFrame(df[['symmetry.crystal_system', 'symmetry.number']]).groupby("symmetry.crystal_system")

map = {}

for group_name, group_data in new_df:
    map[group_name] = group_data["symmetry.number"].drop_duplicates().sort_values().to_numpy()


df = df.drop(columns=['symmetry.point_group', 'formula_pretty'])
df = df.groupby("symmetry.number").count()
df = pd.DataFrame(df[['symmetry.symbol']])
for index, row in df.iterrows():
    print(index)
    for k, v in map.items():
        if index in v:
            df.loc[index, 'crystal_system'] = k
df.to_csv('kind_res.csv', index=True)


