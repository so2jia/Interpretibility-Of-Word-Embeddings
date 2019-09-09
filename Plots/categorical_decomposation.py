import matplotlib.pyplot as plt
import numpy as np

w = np.load("../out/w_nsb.npy")

dim = 2

selected_dim = w[dim, :]

# objects = np.arange(len(selected_dim))
# y_pos = np.arange(len(selected_dim))
#
#
# plt.bar(y_pos, selected_dim, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')
#
# plt.show()