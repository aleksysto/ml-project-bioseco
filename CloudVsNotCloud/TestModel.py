from tensorflow.keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random as r

model = load_model("./66_final_clouds.keras")

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
print(cm)
goodsum = 0
wholesum = 0
goodidx = 0
for arr in cm:
    goodsum += arr[goodidx]
    goodidx += 1
    for elem in arr:
        wholesum += elem

print("Accuracy from confusion matrix: ", goodsum/wholesum)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
