import keras
from keras import layers
from tensorflow import data as tf_data
import random as r
import matplotlib.pyplot as plt


data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.25)
        ]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


image_size = (100, 100)
batch_size = 64

train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "../clouds_data/train",
        validation_split=0.3,
        subset="both",
        seed=r.randint(10000, 20000),
        image_size=image_size,
        batch_size=batch_size,
        )

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# Entry block
inputs = keras.Input(shape=(100, 100, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

previous_block_activation = x  # Set aside residual

for size in [32, 32, 48]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(size, 1, strides=2, padding="same")(
        previous_block_activation
    )
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

x = layers.Conv2D(48, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

x = layers.GlobalAveragePooling2D()(x)
num_classes = 2
if num_classes == 2:
    units = 1
else:
    units = num_classes

x = layers.Dropout(0.25)(x)
# We specify activation=None so as to return logits
outputs = layers.Dense(units, activation=None)(x)


model = keras.Model(inputs, outputs)

epochs = 75

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    verbose=1,
    factor=0.0005,
    min_lr=0.000001
)
save_model_callback = keras.callbacks.ModelCheckpoint(
    filepath="{epoch:02d}_final_clouds.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
callbacks = [learning_rate_reduction, save_model_callback]
loss = keras.losses.BinaryCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(train_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=val_ds)

# model.save("first_try_cloudsVSnotclouds.keras")
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
