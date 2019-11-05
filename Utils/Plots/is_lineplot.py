import seaborn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

paths = [["gauss/kernel_01", "01", "Gauss Kernel 0.1 KDE"],
         ["gauss/kernel_01", "s01", "Gauss Kernel 0.1 KDE Semantic"],
         ["gauss/kernel_02", "02", "Gauss Kernel 0.2 KDE"],
         ["gauss/kernel_02", "s02", "Gauss Kernel 0.2 KDE Semantic"],
         ["gauss/kernel_05", "05", "Gauss Kernel 0.5 KDE"],
         ["gauss/kernel_05", "s05", "Gauss Kernel 0.5 KDE Semantic"],
         ["gauss/kernel_1", "1", "Gauss Kernel 1.0 KDE"],
         ["gauss/kernel_1", "s1", "Gauss Kernel 1.0 KDE Semantic"],
         ["gauss/hellinger", "h", "Hellinger Distance Kernel 0.2"],
         ["gauss/hellinger", "sh", "Hellinger Distance Kernel 0.2 Semantic"],
         ["original", "o", "Closed Bhattacharyya Distance"],
         ["original", "so", "Closed Bhattacharyya Distance Semantic"]]

plot_index = 0

fig, axes = plt.subplots(paths.__len__()//2, 2, sharey=True, sharex=True)
fig.suptitle("Interpretibility")
fig.set_size_inches(8, 18)

semspace = False

for path in paths:

    folder = path[0]
    name_prefix = path[1] + "_"

    files = [[f"w_b_stats.txt", "W_b"],
             # [f"w_nb_stats.txt", "W_nb"],
             # [f"w_nsb_stats.txt", "W_nsb"],
             # [f"w_b_stats-norm.txt", "W_b norm"],
             # [f"w_nb_stats-norm.txt", "W_nb norm"],
             # [f"w_nsb_stats-norm.txt", "W_nsb norm"]
             ]
    prefix = f"../../out/{folder}/results/"

    base_path = os.path.join(os.getcwd(), prefix)

    values = {}
    # Dense
    for file in files:
        if semspace:
            p = os.path.join(base_path, "semantic_"+file[0])
        else:
            p = os.path.join(base_path, file[0])
        with open(p, mode="r", encoding="utf8") as f:
            values[file[1]] = np.array(f.readlines()[3:], dtype=np.float)

    data = pd.DataFrame(data=values, index=[1, 5, 10])

    axs = axes.flat[plot_index]
    ax = seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data, ax=axs)

    ax.set_xlabel("Lambda")
    ax.set_ylabel("IS' (%)")
    ax.set_title(path[2])

    plot_index += 1

    semspace = ~semspace

plt.xticks([1, 5, 10])
plt.xlim([1, 10])

plt.savefig(f"../../out/all_is.png")

plt.show()

# # sparse raw
# for i in range(5):
#     values = {}
#     for path in paths2:
#         p = os.path.join(base_path, path[0].replace("%", str(i+1)))
#         with open(p, mode="r", encoding="utf8") as f:
#             values[path[1]] = np.array(f.readlines()[3:], dtype=np.float)
#     data = pd.DataFrame(data=values, index=np.arange(1, 11, 1))
#     axs = axes.flat[i+1]
#     ax = seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data, ax=axs)
#     ax.set_xlabel("Lambda")
#     ax.set_ylabel("IS' (%)")
#     ax.set_title(f'Sparse embedding space l=0.{i+1}')
#
# # sparse semantic
# for i in range(5):
#     values = {}
#     for path in paths3:
#         p = os.path.join(base_path, path[0].replace("%", str(i+1)))
#         with open(p, mode="r", encoding="utf8") as f:
#             values[path[1]] = np.array(f.readlines()[3:], dtype=np.float)
#     data = pd.DataFrame(data=values, index=np.arange(1, 11, 1))
#     axs = axes.flat[5+i+1]
#     ax = seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data, ax=axs)
#     ax.set_xlabel("Lambda")
#     ax.set_ylabel("IS' (%)")
#     ax.set_title(f'Sparse semantic space l=0.{i+1}')

