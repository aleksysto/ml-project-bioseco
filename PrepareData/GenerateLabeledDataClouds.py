import os
import numpy as np
import shutil


def check_for_element_cloud(array, name, cloud, not_cloud):
    for arr in array:
        file = arr[0].split(".")
        filename = f"{file[0]}.{file[1]}"
        argfile = name.split(".")
        argname = f"{argfile[0]}.{argfile[1]}"
        if filename == argname:
            arr[1] = int(arr[1]) + cloud
            arr[2] = int(arr[2]) + not_cloud
            return True
        return False


clouds_dirs = os.listdir("../clouds_confirm/")

clouds_files = []

for dir in clouds_dirs:
    files = os.listdir(f"../clouds_confirm/{dir}")
    for file in files:
        clouds_files.append(file)
        shutil.copyfile(
            f"../clouds_confirm/{dir}/{file}",
            f"../all_clouds/{file}"
        )
cloud_array = np.empty([0, 3])
print(cloud_array)
for file in clouds_files:
    file_attributes = file.split(".")
    filename = file
    label = file_attributes[2]
    cloud = 0
    not_cloud = 0
    if label == "cloud":
        cloud = 1
    elif label == "not_cloud":
        not_cloud = 1
    check = check_for_element_cloud(cloud_array, filename, cloud, not_cloud)
    if not check:
        cloud_array = np.append(cloud_array, [[filename, cloud, not_cloud]], axis=0)

for arr in cloud_array:
    file = arr[0].split(".")
    filename = f"{file[0]}.{file[1]}"
    if arr[1] > arr[2]:
        shutil.copyfile(f"../all_clouds/{arr[0]}", f"../labeled_clouds/{filename}.cloud.{file[-1]}")
    elif arr[2] > arr[1]:
        shutil.copyfile(f"../all_clouds/{arr[0]}", f"../labeled_clouds/{filename}.not_cloud.{file[-1]}")
    else:
        continue

# wrzucic z planes_assign do cloudow
# dla planes klasa wielokolumnowa i procenty
# jako dane wejsciowe id_plane_bird.jpg
# CSV z id/nazwapliku i kolumny
# na wyjsciu 8 neuronow
# warto zrobic eksperyment dla rekordow gdzie jedna osoba oznaczala bo moga byc gorsze niz jak wiele osob
# 
