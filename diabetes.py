import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

img_height, img_width = 224, 224
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/train",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/validation",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = ["diabete", "nodiabete"]

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

model.evaluate(test_ds)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_index = np.argmax(predictions[i])
        plt.title(f"Pred: {class_names[pred_index]} | Real: {class_names[labels[i]]}")
        plt.axis("off")
plt.show()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

with open("labels.txt", "w") as f:
    f.write("0 diabete\n1 nodiabete")
