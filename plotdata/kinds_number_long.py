import csv
import json
import os
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# 创建 DataFrame
df = pd.DataFrame(columns=["formula_pretty", "symmetry.number", "symmetry.crystal_system", "symmetry.symbol", "symmetry.point_group"])

base_dir = "../data/materials_datas/{}.json"
res = {}
for i in range(1, 231):
    res[i] = 0

for i in range(13):
    print(i)
    dir = base_dir.format(i)
    with open(dir, 'r') as file:
        data = json.load(file)

        for k, v in tqdm(data.items()):
            new_row = [v["formula_pretty"],
                       v["symmetry.number"],
                       v["symmetry.crystal_system"],
                       v["symmetry.symbol"],
                       v["symmetry.point_group"]]
            df = df.append(pd.Series(new_row, index=df.columns), ignore_index=True)

df.to_csv('data.csv', index=False)

