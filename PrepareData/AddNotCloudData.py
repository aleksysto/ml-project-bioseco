import os
import pandas as pd
import numpy as np
import shutil


index = 0
for file in os.listdir("../labeled_planes"):
    shutil.copyfile(
        f"../labeled_planes/{file}",
        f"../clouds_data/not_cloud.{index}.jpg"
    )
    index += 1

for file in os.listdir("../labeled_clouds"):
    label = file.split(".")[2]
    shutil.copyfile(
            f"../labeled_clouds/{file}",
            f"../clouds_data/{label}.{index}.jpg"
            )
    index += 1
