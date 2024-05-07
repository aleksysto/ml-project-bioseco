from tensorflow.keras.models import load_model
import keras
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import random as r
import os

model = keras.models.load_model("./65_final_clouds.keras")

image_size = (100, 100)

clouds_array = []
clouds_path_array = []
for dir in os.listdir("../clouds_data/train/"):
    images = os.listdir(f"../clouds_data/train/{dir}")
    for i in range(4):
        random_number = r.randint(0, len(images) - 1)
        image_path = f"../clouds_data/train/{dir}/{images[random_number]}"
        loaded_img = keras.utils.load_img(image_path,
                                          color_mode="rgb",
                                          target_size=image_size)
        clouds_array.append(loaded_img)
        clouds_path_array.append(image_path)

labels = [0, 0, 0, 0, 1, 1, 1, 1]
predictions = []
id = 0
for image in clouds_array:
    array = keras.utils.img_to_array(image)
    batch = np.array([array])
    prediction2 = model.predict(batch)
    predictions.append(prediction2)

#predict_labels = []
#for prediction in predictions:
#    if prediction[0] > 0:
#        predict_labels.append(1)
#    elif prediction[0] < 0:
#        predict_labels.append(0)
#    else:
#        predict_labels.append(0)

print(labels)
print(predictions)

loaded_images = []
for path in clouds_path_array:
    load_img = np.asarray(PIL.Image.open(path))
    loaded_images.append(load_img)
id = 1
fig = plt.figure(figsize=(10, 7))
for img in loaded_images:
    if id == 5:
        break
    fig.add_subplot(2, 2, id)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True label: {labels[id-1]}, Predicted label: {predictions[id-1]}")
    id += 1

plt.show()

fig = plt.figure(figsize=(10, 7))
id = 1
for i in range(4):
    fig.add_subplot(2, 2, id)
    plt.imshow(loaded_images[i+4])
    plt.axis("off")
    plt.title(f"True label: {labels[i+4]}, Predicted label: {predictions[i+4]}")
    id += 1

plt.show()
