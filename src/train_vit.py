import json

import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

from src.normalization_layers import *


def train_vit(dataset, logdir, num_epochs=50, order='l2', radius='ur', run_exp_count=1):
    dataset_name = dataset["name"]

    train_ds, validation_ds = tfds.load(
        dataset_name,
        split=[dataset["train_split"], dataset["validation_split"]],
        as_supervised=True
    )

    num_classes = dataset["num_classes"]
    IMG_SHAPE = dataset["IMG_SHAPE"]
    batch_size = dataset["batch_size"]
    learning_rate = dataset["learning_rate"]
    input_shape = (IMG_SHAPE, IMG_SHAPE, 3)
    patch_size = 6  # Size of the patches to be extract from the input images
    image_size = 72  # We'll resize input images to this size
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim, ]  # Size of the transformer layers
    transformer_layers = 8

    def preprocess_image(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))

        return (image, label)

    train_ds = train_ds.map(
        preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    validation_ds = validation_ds.map(
        preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.cache()
    validation_ds = validation_ds.batch(batch_size)
    validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Add Data Augmentation pipeline
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.02),
            layers.experimental.preprocessing.RandomWidth(0.2),
            layers.experimental.preprocessing.RandomHeight(0.2),
        ]
    )
    data_augmentation.layers[0].adapt(next(iter(train_ds))[0])

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super(Patches, self).__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super(PatchEncoder, self).__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    def create_classifier():
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        features = layers.Flatten()(representation)

        if not (order == 'nl' and radius == 'nr'):
            normalization = get_normalization_layer(order, radius)
            features = normalization(features)

        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        return model

    # repeat the exp for all counts
    for run_exp in range(run_exp_count):
        classifier = create_classifier()
        train_history = classifier.fit(train_ds, batch_size=batch_size, epochs=num_epochs,
                                       validation_data=validation_ds)

        # log training history
        json.dump(train_history.history, open(f"{logdir}/th_vit_{order}_{radius}_run{run_exp}.json", 'w'), indent=4)
