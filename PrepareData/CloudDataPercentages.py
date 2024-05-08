import os
import matplotlib.pyplot as plt

classes = {"cloud": 0,
           "not_cloud": 0,
           }
arg_count = 0
for file in os.listdir("../labeled_clouds/"):
    row = file.split(".")[2]
    classes[row] += 1


names = list(classes.keys())
values = list(classes.values())

plt.bar(range(len(classes)), values, tick_label=names)
plt.show()
