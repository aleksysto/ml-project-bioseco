import os
import shutil
import matplotlib.pyplot as plt


index = 0
classes = {"not_cloud": 0,
           "cloud": 0}

for file in os.listdir("../clouds_data"):
    label = file.split(".")[0]
    if label == "train" or label == "test":
        continue
    index += 1
    classes[label] += 1

names = list(classes.keys())
values = list(classes.values())

plt.bar(range(len(classes)), values, tick_label=names)
plt.show()

not_clouds_count = 0
clouds_count = 0

clouds_data_files = os.listdir("../clouds_data")
index = 0

for file in os.listdir("../clouds_data/"):
    if file == "test" or file == "train":
        continue
    shutil.move(f"../clouds_data/{file}", f"../clouds_data/train/{file}")

for file in os.listdir("../clouds_data/train/"):
    if file == "cloud" or file == "not_cloud":
        continue
    filename = file.split(".")[0]
    shutil.move(f"../clouds_data/train/{file}", f"../clouds_data/train/{filename}/{file}")
