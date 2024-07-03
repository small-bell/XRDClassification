import csv
import json
import os
import numpy as np


base_dir = "../data/materials_datas/{}.json"
res = {}
for i in range(1, 231):
    res[i] = 0

for i in range(13):
    dir = base_dir.format(i)
    with open(dir, 'r') as file:
        data = json.load(file)

        for k, v in data.items():
            num = int(v["symmetry.number"])
            res[num] = res[num] + 1




csv_data = []
for k, v in res.items():
    csv_data.append([str(k), str(v)])

print(csv_data)

with open("csv_file.csv", 'w') as file:
    writer = csv.writer(file)
    for line in csv_data:
        print(line)
        writer.writerow(line)
