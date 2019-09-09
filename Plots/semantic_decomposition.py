import matplotlib.pyplot as plt
import numpy as np
from Utils.Loaders.semcat import read

plt.rcdefaults()
fig, ax = plt.subplots()

lines = []
word = "article"
y_axis_label = []
x_value = []
in_cat = []

semcat = read("../data/semcat/Categories/")

with open(f"../out/decom-{word}.txt", mode="r", encoding="utf8") as f:
    for line in f.readlines():
        l = line.split(' ')
        lines.append(l)
        y_axis_label.append(l[0])
        x_value.append(float(l[2]))
        if word in semcat.vocab[l[0]]:
            in_cat.append(1)
        else:
            in_cat.append(0)




# Example data
y_pos = np.arange(len(y_axis_label))


barlist = ax.barh(y_pos, x_value, align='center')

for i, v in enumerate(in_cat):
    if v == 1:
        barlist[i].set_color('r')


ax.set_yticks(y_pos)
ax.set_yticklabels(y_axis_label)
ax.invert_yaxis()
ax.set_xlabel('')
ax.set_title(f'Semantical decomposition of word {word}')

plt.show()