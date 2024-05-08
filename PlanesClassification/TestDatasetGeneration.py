import os
import shutil
import pandas as pd
from PIL import Image
import keras
from keras import layers
import random as r

label_csv = pd.read_csv("./planes.csv")
print(label_csv)
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2)
]


def max2(arr):
    max = arr[0]
    index = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            index = i
    return max, index


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


not_plane = []
plane_common = []
plane_large = []
plane_with_trail = []
plane_too_far = []
plane_partial = []
helicopter = []
for file in sorted(os.listdir("../labeled_planes/")):
    row = label_csv.loc[label_csv["file"] == file]
    row_labels = row.iloc[:, 2:10]
    labels = row_labels.values[0]
    labels_as_arr = [x for x in labels]
    max, index = max2(labels_as_arr)
    match index:
        case 0:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/not_plane/{file}")
            not_plane.append(file)
        case 1:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/plane_common/{file}")
            plane_common.append(file)
        case 2:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/plane_large/{file}")
            plane_large.append(file)
        case 3:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/plane_with_trail/{file}")
            plane_with_trail.append(file)
        case 4:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/plane_too_far/{file}")
            plane_too_far.append(file)
        case 5:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/plane_partial/{file}")
            plane_partial.append(file)
        case 6:
            shutil.copy(f"../labeled_planes/{file}", f"../planes_data/helicopter/{file}")
            helicopter.append(file)
        case 7:
            pass
arrays = [not_plane, plane_common, plane_large, plane_with_trail, plane_too_far, plane_partial, helicopter]
for arr in arrays:
    print(len(arr))
rotations_arr = [90, 180, 270]
while len(plane_with_trail) > 4500:
    random_id = r.randint(0, len(plane_with_trail) - 1)
    filename = plane_with_trail[random_id]
    os.remove(f"../planes_data/plane_with_trail/{filename}")
    plane_with_trail.remove(filename)

index = 0
while len(plane_common) < 4500:
    index += 1
    filename = plane_common[r.randint(0, len(plane_common) - 1)]
    filename_index = filename.split(".")[0]
    original = Image.open(f"../planes_data/plane_common/{filename}")
    rotated = original.rotate(rotations_arr[r.randint(0, 2)])
    saved = rotated.save(f"../planes_data/plane_common/{filename_index}_{index}.jpg")
    plane_common.append(f"{filename_index}_{index}.jpg")
index = 0
while len(plane_large) < 4500:
    index += 1
    filename = plane_large[r.randint(0, len(plane_large) - 1)]
    filename_index = filename.split(".")[0]
    original = Image.open(f"../planes_data/plane_large/{filename}")
    rotated = original.rotate(rotations_arr[r.randint(0, 2)])
    saved = rotated.save(f"../planes_data/plane_large/{filename_index}_{index}.jpg")
    plane_large.append(f"{filename_index}_{index}.jpg")
index = 0
while len(plane_too_far) < 4500:
    index += 1
    filename = plane_too_far[r.randint(0, len(plane_too_far) - 1)]
    filename_index = filename.split(".")[0]
    original = Image.open(f"../planes_data/plane_too_far/{filename}")
    rotated = original.rotate(rotations_arr[r.randint(0, 2)])
    saved = rotated.save(f"../planes_data/plane_too_far/{filename_index}_{index}.jpg")
    plane_too_far.append(f"{filename_index}_{index}.jpg")
index = 0
while len(plane_partial) < 4500:
    index += 1
    filename = plane_partial[r.randint(0, len(plane_partial) - 1)]
    filename_index = filename.split(".")[0]
    original = Image.open(f"../planes_data/plane_partial/{filename}")
    rotated = original.rotate(rotations_arr[r.randint(0, 2)])
    saved = rotated.save(f"../planes_data/plane_partial/{filename_index}_{index}.jpg")
    plane_partial.append(f"{filename_index}_{index}.jpg")
index = 0
while len(helicopter) < 4500:
    index += 1
    filename = helicopter[r.randint(0, len(helicopter) - 1)]
    filename_index = filename.split(".")[0]
    original = Image.open(f"../planes_data/helicopter/{filename}")
    rotated = original.rotate(rotations_arr[r.randint(0, 2)])
    saved = rotated.save(f"../planes_data/helicopter/{filename_index}_{index}.jpg")
    helicopter.append(f"{filename_index}_{index}.jpg")


image_size = (100, 100)
batch_size = 128
train_ds = keras.utils.image_dataset_from_directory(
    "../planes_data/",
    label_mode="categorical",
    seed=r.randint(10000, 20000),
    image_size=image_size,
    batch_size=batch_size,
)
