import seaborn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< Updated upstream
<<<<<<< Updated upstream
paths = [["kde/dense_results/dense_w_b_10-stats.txt", "W_b"],
         ["kde/dense_results/dense_w_nb_10-stats.txt", "W_nb"],
         ["kde/dense_results/dense_w_nsb_10-stats.txt", "W_nsb"],
         ["kde/dense_results/dense_w_b_10-norm-stats.txt", "W_b norm"],
         ["kde/dense_results/dense_w_nb_10-norm-stats.txt", "W_nb norm"],
         ["kde/dense_results/dense_w_nsb_10-norm-stats.txt", "W_nsb norm"]]
=======
=======
>>>>>>> Stashed changes
name_prefix = "dense_"

paths = [[f"{name_prefix}w_b_10-stats.txt", "W_b"],
         [f"{name_prefix}w_nb_10-stats.txt", "W_nb"],
         [f"{name_prefix}w_nsb_10-stats.txt", "W_nsb"],
         [f"{name_prefix}w_b_10-norm-stats.txt", "W_b norm"],
         [f"{name_prefix}w_nb_10-norm-stats.txt", "W_nb norm"],
         [f"{name_prefix}w_nsb_10-norm-stats.txt", "W_nsb norm"]]
<<<<<<< Updated upstream
>>>>>>> Stashed changes

paths2 = [["kde/sparse_results/raw/l0%/l0%_w_b_10-stats.txt", "W_b"],
          ["kde/sparse_results/raw/l0%/l0%_w_nb_10-stats.txt", "W_nb"],
          ["kde/sparse_results/raw/l0%/l0%_w_nsb_10-stats.txt", "W_nsb"],
          ["kde/sparse_results/raw/l0%/l0%_w_b_10-norm-stats.txt", "W_b norm"],
          ["kde/sparse_results/raw/l0%/l0%_w_nb_10-norm-stats.txt", "W_nb norm"],
          ["kde/sparse_results/raw/l0%/l0%_w_nsb_10-norm-stats.txt", "W_nsb norm"]]

paths3 = [["kde/sparse_results/semantic/l0%/l_0.%_w_b_10-stats.txt", "W_b"],
          ["kde/sparse_results/semantic/l0%/l_0.%_w_nb_10-stats.txt", "W_nb"],
          ["kde/sparse_results/semantic/l0%/l_0.%_w_nsb_10-stats.txt", "W_nsb"],
          ["kde/sparse_results/semantic/l0%/l_0.%_w_b_10-norm-stats.txt", "W_b norm"],
          ["kde/sparse_results/semantic/l0%/l_0.%_w_nb_10-norm-stats.txt", "W_nb norm"],
          ["kde/sparse_results/semantic/l0%/l_0.%_w_nsb_10-norm-stats.txt", "W_nsb norm"]]

prefix = "../../out/"
=======

# paths2 = [["kde/sparse_results/raw/l0%/l0%_w_b_10-stats.txt", "W_b"],
#           ["kde/sparse_results/raw/l0%/l0%_w_nb_10-stats.txt", "W_nb"],
#           ["kde/sparse_results/raw/l0%/l0%_w_nsb_10-stats.txt", "W_nsb"],
#           ["kde/sparse_results/raw/l0%/l0%_w_b_10-norm-stats.txt", "W_b norm"],
#           ["kde/sparse_results/raw/l0%/l0%_w_nb_10-norm-stats.txt", "W_nb norm"],
#           ["kde/sparse_results/raw/l0%/l0%_w_nsb_10-norm-stats.txt", "W_nsb norm"]]
#
# paths3 = [["kde/sparse_results/semantic/l0%/l_0.%_w_b_10-stats.txt", "W_b"],
#           ["kde/sparse_results/semantic/l0%/l_0.%_w_nb_10-stats.txt", "W_nb"],
#           ["kde/sparse_results/semantic/l0%/l_0.%_w_nsb_10-stats.txt", "W_nsb"],
#           ["kde/sparse_results/semantic/l0%/l_0.%_w_b_10-norm-stats.txt", "W_b norm"],
#           ["kde/sparse_results/semantic/l0%/l_0.%_w_nb_10-norm-stats.txt", "W_nb norm"],
#           ["kde/sparse_results/semantic/l0%/l_0.%_w_nsb_10-norm-stats.txt", "W_nsb norm"]]

prefix = "../../out/exponential/result/"
>>>>>>> Stashed changes

base_path = os.path.join(os.getcwd(), prefix)

values = {}
# Dense
for path in paths:
    p = os.path.join(base_path, path[0])
    with open(p, mode="r", encoding="utf8") as f:
        values[path[1]] = np.array(f.readlines()[3:], dtype=np.float)

data = pd.DataFrame(data=values, index=np.arange(1, 11, 1))

fig, axes = plt.subplots(3, 4, sharey=True, sharex=True)
fig.suptitle("Interpretibility")
fig.set_size_inches(20, 15)


ax = seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data, ax=axes.flat[0])
ax.set_xlabel("Lambda")
ax.set_ylabel("IS' (%)")
<<<<<<< Updated upstream
<<<<<<< Updated upstream
ax.set_title('Dense embedding space')

# sparse raw
for i in range(5):
    values = {}
    for path in paths2:
        p = os.path.join(base_path, path[0].replace("%", str(i+1)))
        with open(p, mode="r", encoding="utf8") as f:
            values[path[1]] = np.array(f.readlines()[3:], dtype=np.float)
    data = pd.DataFrame(data=values, index=np.arange(1, 11, 1))
    axs = axes.flat[i+1]
    ax = seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data, ax=axs)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("IS' (%)")
    ax.set_title(f'Sparse embedding space l=0.{i+1}')
=======
ax.set_title('Dense (Exponential))')
>>>>>>> Stashed changes

# sparse semantic
for i in range(5):
    values = {}
    for path in paths3:
        p = os.path.join(base_path, path[0].replace("%", str(i+1)))
        with open(p, mode="r", encoding="utf8") as f:
            values[path[1]] = np.array(f.readlines()[3:], dtype=np.float)
    data = pd.DataFrame(data=values, index=np.arange(1, 11, 1))
    axs = axes.flat[5+i+1]
    ax = seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data, ax=axs)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("IS' (%)")
    ax.set_title(f'Sparse semantic space l=0.{i+1}')
=======
ax.set_title('Dense (Exponential))')

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
>>>>>>> Stashed changes

plt.xticks(np.arange(1, 11, 1))
plt.xlim([1, 10])

plt.savefig("../../out/kde/is.png")

plt.show()