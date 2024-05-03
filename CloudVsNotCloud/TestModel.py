from tensorflow.keras.models import load_model
import keras
import numpy as np
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

test1_it, test_it = keras.utils.image_dataset_from_directory(
        "../clouds_data/train",
        validation_split=0.3,
        subset="both",
        seed=r.randint(10000, 20000),
        image_size=image_size,
        batch_size=batch_size,
        )
print(test_it)
labels = np.concatenate([y for x, y in test_it], axis=0)
predictions = model.predict(test_it, steps=len(test_it))
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

