import os
files = os.listdir("../planes_data/plane_with_trail/")

#while len(files) > 4500:
#    index = r.randint(0, len(files)-1)
#    os.remove(f"../planes_data/plane_with_trail/{files[index]}")
#    files.remove(files[index])
sizes = []
for folder in os.listdir("../planes_data/"):
    sizes.append(len(os.listdir(f"../planes_data/{folder}")))

print(sizes)
