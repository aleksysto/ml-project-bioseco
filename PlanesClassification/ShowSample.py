from tensorflow.keras.models import load_model
import keras
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random as r

model = load_model("./new_data_model/73_new_data_second_go.keras")

image_size = (100, 100)
images_array = []
images_path_array = []
for dir in os.listdir("../planes_data/"):
    counter = 0
    for i in range(4):
        images = os.listdir(f"../planes_data/{dir}")
        image = images[r.randint(0, len(images)-1)]
        print(f"../planes_data/{dir}/{image}")
        loaded_img = keras.utils.load_img(f"../planes_data/{dir}/{image}",
                                          color_mode="rgb",
                                          target_size=image_size)
        images_path_array.append(f"../planes_data/{dir}/{image}")
        images_array.append(loaded_img)
        counter += 1
print(images_path_array)
predictions = []
for image in images_array:
    array = keras.utils.img_to_array(image)
    batch = np.array([array])
    prediction2 = model.predict(batch)
    predictions.append(prediction2)
print(predictions)

show_images = []
for path in images_path_array:
    loaded = np.asarray(PIL.Image.open(path))
    show_images.append([loaded, path])

id = 1
id = 1
fig = plt.figure(figsize=(26, 7))
for index, loaded_array in enumerate(show_images):
    if id > 4:
        plt.show()
        id = 1
        fig = plt.figure(figsize=(26, 7))
    fig.add_subplot(2, 2, id)
    plt.imshow(loaded_array[0])
    path = loaded_array[1]
    label = path.split("/")[2]
    plt.axis("off")
    plt.title(f"True label: {label},\nPredicted label: {predictions[index]}")
    id += 1
plt.show()
