import matplotlib.pyplot as plt
import numpy as np

scores = []

with open("../out/scores.txt", mode="r", encoding="utf8") as f:
    for l in f.readlines():
        scores.append(float(l))

lamb = np.arange(len(scores))

fig, ax = plt.subplots()
ax.plot(lamb, scores)

ax.set(xlabel='Lambda', ylabel='Score (%)',
       title='Interpretability Score')
ax.set_xticks(lamb)
ax.grid()

plt.show()