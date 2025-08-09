import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Dataset
#Data Manipulation
img_height, img_width = 224, 224
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/validation",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
#Data Visualization
class_names = ["diabete","nodiabete"]
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()


# Step 2: Model
model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./127.5),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(2)
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

# Step 3: Train
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 50
)
#Summary of the model
model.summary()
# Step 4: Test
model.evaluate(test_ds)


