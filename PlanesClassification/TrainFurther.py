import keras
from keras import layers
from tensorflow import data as tf_data
import random as r
import matplotlib.pyplot as plt

data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.35)
        ]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


image_size = (100, 100)
batch_size = 16
train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "../planes_data/",
        validation_split=0.38,
        label_mode="categorical",
        shuffle=True,
        subset="both",
        seed=r.randint(10000, 20000),
        image_size=image_size,
        batch_size=batch_size,
        )

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

model = keras.models.load_model("./new_data_model/73_new_data_second_go.keras")

epochs = 75

optimizer = keras.optimizers.Adam(learning_rate=0.000001)
optimizer2 = keras.optimizers.SGD(learning_rate=0.01)
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    verbose=1,
    factor=0.005,
    min_lr=0.000001
)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath="./new_data_model/{epoch:02d}_new_data_third_go.keras",
            save_best_only=True,
            monitor="val_accuracy",
            mode='max',
        )
callbacks = [learning_rate_reduction, checkpoint_callback]
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)

model.save("new_data_model/model_third.keras")
# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()
