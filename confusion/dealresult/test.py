import pandas as pd

# 创建示例DataFrame
df = pd.read_csv('all-model.csv')

df.columns = [str(i) for i in range(17)]


df = pd.concat([df["1"], df["3"]], axis=1)

print(df.shape)
df.to_csv("all-confusion.csv", header=False, index=False)

