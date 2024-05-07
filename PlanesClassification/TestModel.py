from tensorflow.keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random as r

def find_label(arr):
    max = arr[0]
    id = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            id = i
    return id

model = load_model("./new_data_model/56_new_data_third_go.keras")
model1 = load_model("./new_data_model/73_new_data_second_go.keras")
model2 = load_model("./new_data_model/75_new_data.keras")

image_size = (100, 100)
batch_size = 128

test1_it, test_it = keras.utils.image_dataset_from_directory(
        "../planes_data/",
        validation_split=0.85,
        subset="both",
        seed=r.randint(10000, 20000),
        image_size=image_size,
        batch_size=batch_size,
        )

labels = np.concatenate([y for x, y in test_it], axis=0)
for x, y in test_it:
    print(x)
    print(y)
predictions = model.predict(test_it, steps=len(test_it))
predict_labels = []
for predict in predictions:
    label = find_label(predict)
    predict_labels.append(label)

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

predictions = model1.predict(test_it, steps=len(test_it))
predict_labels = []
for predict in predictions:
    label = find_label(predict)
    predict_labels.append(label)

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
predictions = model2.predict(test_it, steps=len(test_it))
predict_labels = []
for predict in predictions:
    label = find_label(predict)
    predict_labels.append(label)

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
