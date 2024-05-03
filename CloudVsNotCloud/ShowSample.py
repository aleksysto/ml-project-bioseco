from tensorflow.keras.models import load_model
import keras
import PIL
import PIL.Image
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random as r

FAST_RUN = False
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
epochs = 1


model = load_model("./second_try_cloudsVSnotclouds.keras")

image_size = (100, 100)
batch_size = 128
cloud1 = keras.utils.load_img("../clouds_data/train/cloud/cloud.17873.jpg", color_mode="rgb", target_size=(100, 100), keep_aspect_ratio=True)
cloud2 = keras.utils.load_img("../clouds_data/train/cloud/cloud.17898.jpg", color_mode="rgb", target_size=(100, 100))
notcloud1 = keras.utils.load_img("../clouds_data/train/not_cloud/not_cloud.57.jpg", color_mode="rgb", target_size=(100, 100))
notcloud2 = keras.utils.load_img("../clouds_data/train/not_cloud/not_cloud.441.jpg", color_mode="rgb", target_size=(100, 100))
labels = [0, 0, 1, 1]
test_it = [cloud1, cloud2, notcloud1, notcloud2]
predictions = []
id = 0
for image in test_it:
    array = keras.utils.img_to_array(image)
    batch = np.array([array])
    prediction2 = model.predict(batch)
    predictions.append(prediction2)

predict_labels = []
for prediction in predictions:
    if prediction[0] > 0:
        predict_labels.append(1)
    elif prediction[0] < 0:
        predict_labels.append(0)
    else:
        predict_labels.append(0)

print(labels)
print(predict_labels)

# Confusion matrix
cm = confusion_matrix(labels, predict_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

img1 = np.asarray(PIL.Image.open("../clouds_data/train/cloud/cloud.17873.jpg"))
img2 = np.asarray(PIL.Image.open("../clouds_data/train/cloud/cloud.17898.jpg"))
img3 = np.asarray(PIL.Image.open("../clouds_data/train/not_cloud/not_cloud.57.jpg"))
img4 = np.asarray(PIL.Image.open("../clouds_data/train/not_cloud/not_cloud.441.jpg"))
img_arr = [img1, img2, img3, img4]
id = 1
fig = plt.figure(figsize=(10, 7))
for img in img_arr:
    fig.add_subplot(2, 2, id)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True label: {labels[id-1]}, Predicted label: {predict_labels[id-1]}")
    id += 1

plt.show()
