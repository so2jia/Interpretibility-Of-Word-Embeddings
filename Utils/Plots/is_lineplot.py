import seaborn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

paths = [["cor_val_interpret_w_b_10-stats.txt", "W_b"],
         ["cor_val_interpret_w_nb_10-stats.txt", "W_nb"],
         ["cor_val_interpret_w_nsb_10-stats.txt", "W_nsb"],
         ["cor_val_interpret_w_b_10-norm-stats.txt", "W_b norm"],
         ["cor_val_interpret_w_nb_10-norm-stats.txt", "W_nb norm"],
         ["cor_val_interpret_w_nsb_10-norm-stats.txt", "W_nsb norm"]]

prefix = "../../out/I_correlation_tests"

base_path = os.path.join(os.getcwd(), prefix)

values = {}


for path in paths:
    p = os.path.join(base_path, path[0])
    with open(p, mode="r", encoding="utf8") as f:
        values[path[1]] = np.array(f.readlines()[3:], dtype=np.float)

data = pd.DataFrame(data=values, index=np.arange(1, 11, 1))

seaborn.lineplot(hue="Mathod", markers=True, dashes=False, data=data)
plt.xlabel("Lambda")
plt.ylabel("IS' (%)")
plt.title("Interpretability")
plt.xticks(np.arange(1, 11, 1))
plt.xlim([1, 10])
plt.show()

print(data)