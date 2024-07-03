import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial']

dpi = 1200

# 计算图像尺寸 (英寸)
width_inch = 7480 / dpi
height_inch = 6000 / dpi
width_inch = 3600 / dpi
height_inch = 3000 / dpi
plt.figure(figsize=(width_inch, height_inch))

df = pd.read_csv('all-confusion.csv')
y_true = df.iloc[:, 0].to_numpy()
y_pred = df.iloc[:, 1].to_numpy()


cm = confusion_matrix(y_true, y_pred)
cm = cm / cm.sum(axis=1, keepdims=True)
print(type(cm))
x_labels = ["P1", "P-1", "P2", "Pm", "Pc", "P2/m", "P2/c", "P2_1/m", "P2_1/c", "C2/m"]
y_labels = ["P1", "P-1", "P2", "Pm", "Pc", "P2/m", "P2/c", "P2_1/m", "P2_1/c", "C2/m"]
sns.set_style("white")
sns.despine(offset=10, top=False, right=False, left=False, bottom=False)
ax = sns.heatmap(cm, cmap='OrRd',
# ax = sns.heatmap(cm, cmap='hot_r',
                 linewidths=0.1,
                 linecolor='white',
                 square=True,
                 annot=False,
                 xticklabels=x_labels,
                 yticklabels=y_labels)

for _, spine in ax.spines.items():
    spine.set_visible(True)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

# ax.spines['top'].set_linewidth(100)
# ax.spines['right'].set_linewidth(100)
# ax.spines['bottom'].set_linewidth(100)
# ax.spines['left'].set_linewidth(100)

plt.xlabel("Predicted")
plt.ylabel("True")
# plt.title("Normalized Confusion Matrix")

plt.tight_layout()

import io
from PIL import Image

# png1 = io.BytesIO()
# plt.savefig(png1, format="png", dpi=1200, pad_inches=.1, bbox_inches='tight')
# png2 = Image.open(png1)
# png2.save("ap_qt_ad.tiff")
# png1.close()
#
# plt.show()
plt.savefig('test.tif',dpi=1200,pil_kwargs={"compression": "tiff_lzw"})



