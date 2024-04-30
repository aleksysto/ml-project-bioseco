import os
import pandas as pd
import numpy as np
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

print("not clouds: ", 0.3*classes["not_cloud"], "\n\n clouds: ", 0.3*classes["cloud"])

plt.bar(range(len(classes)), values, tick_label=names)
plt.show()

not_clouds_count = 0
not_clouds_test_dataset_count = 0.3 * classes["not_cloud"]
clouds_count = 0
clouds_test_dataset_count = 0.3 * classes["cloud"]

clouds_data_files = os.listdir("../clouds_data")
index = 0
while not_clouds_count < not_clouds_test_dataset_count:
    file = clouds_data_files[index]
    label = file.split(".")[0]
    if label == "not_cloud":
        shutil.move(f"../clouds_data/{file}", f"../clouds_data/test/{file}")
    index += 1
    not_clouds_count += 1

index = 0
while clouds_count < clouds_test_dataset_count:
    file = clouds_data_files[index]
    label = file.split(".")[0]
    if label == "cloud":
        shutil.move(f"../clouds_data/{file}", f"../clouds_data/test/{file}")
    index += 1
    clouds_count += 1

for file in os.listdir("../clouds_data/"):
    if file == "test" or file == "train":
        continue
    shutil.move(f"../clouds_data/{file}", f"../clouds_data/train/{file}")
