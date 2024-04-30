import os
import pandas as pd
import numpy as np
import shutil


def check_for_element_plane(array, name, label_dict):
    for arr in array:
        file = arr[0].split(".")
        filename = f"{file[0]}.{file[1]}"
        argfile = name.split(".")
        argname = f"{argfile[0]}.{argfile[1]}"
        if filename == argname:
            arr[1] = int(arr[1]) + label_dict["not_plane"]
            arr[2] = int(arr[2]) + label_dict["plane_common"]
            arr[3] = int(arr[3]) + label_dict["plane_large"]
            arr[4] = int(arr[4]) + label_dict["plane_with_trail"]
            arr[5] = int(arr[5]) + label_dict["plane_too_far"]
            arr[6] = int(arr[6]) + label_dict["plane_partial"]
            arr[7] = int(arr[7]) + label_dict["helicopter"]
            arr[8] = int(arr[8]) + label_dict["something_strange"]
            return True
    return False


planes_dirs = os.listdir("../planes_assign/")

planes_files = []

for dir in planes_dirs:
    files = os.listdir(f"../planes_assign/{dir}")
    for file in files:
        if file.split(".")[2] != "not_assigned":
            planes_files.append(file)
            shutil.copyfile(
                f"../planes_assign/{dir}/{file}",
                f"../all_planes/{file}"
            )
print(planes_files)
planes_array = np.empty([0, 9])
print(planes_array)
for file in planes_files:
    file_attributes = file.split(".")
    filename = file
    label = file_attributes[2]
    label_dict = {"not_plane": 0,
                  "plane_common": 0,
                  "plane_large": 0,
                  "plane_with_trail": 0,
                  "plane_too_far": 0,
                  "plane_partial": 0,
                  "helicopter": 0,
                  "something_strange": 0
                  }
    # wtf sometimes is not_assigned
    label_dict[label] += 1
    label_array = [filename]
    for key, value in label_dict.items():
        label_array.append(value)
    check = check_for_element_plane(planes_array, filename, label_dict)
    if not check:
        planes_array = np.append(planes_array,
                                 [label_array],
                                 axis=0)

df = pd.DataFrame(columns=["file",
                           "not_plane",
                           "plane_common",
                           "plane_large",
                           "plane_with_trail",
                           "plane_too_far",
                           "plane_partial",
                           "helicopter",
                           "something_strange"])
ind = 0
for arr in planes_array:
    print(arr)
    file = arr[0].split(".")
    filename = f"{file[0]}.{file[1]}"
    series = [f"{ind}.jpg",
              arr[1],
              arr[2],
              arr[3],
              arr[4],
              arr[5],
              arr[6],
              arr[7],
              arr[8]]
    df.loc[len(df.index)] = series
    shutil.copyfile(
        f"../all_planes/{arr[0]}", f"../labeled_planes/{ind}.{file[-1]}")
    ind += 1
print(df)
df.to_csv("./planes.csv", sep=",")

# wrzucic z planes_assign do cloudow
# dla planes klasa wielokolumnowa i procenty
# jako dane wejsciowe id_plane_bird.
# CSV z id/nazwapliku i kolumny
# na wyjsciu 8 neuronow
# warto zrobic eksperyment dla rekordow gdzie jedna osoba oznaczala bo moga byc gorsze niz jak wiele osob
#
