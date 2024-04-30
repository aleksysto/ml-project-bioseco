import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt

df = pd.read_csv("./planes.csv")
max_per_row = df[["not_plane",
                  "plane_common",
                  "plane_large",
                  "plane_with_trail",
                  "plane_too_far",
                  "plane_partial",
                  "helicopter",
                  "something_strange"]].idxmax(axis=1)

classes = {"not_plane": 0,
           "plane_common": 0,
           "plane_large": 0,
           "plane_with_trail": 0,
           "plane_too_far": 0,
           "plane_partial": 0,
           "helicopter": 0,
           "something_strange": 0}
arg_count = 0
for row in max_per_row:
    arg_count += 1
    print(row)
    classes[row] += 1


names = list(classes.keys())
values = list(classes.values())

plt.bar(range(len(classes)), values, tick_label=names)
plt.show()
