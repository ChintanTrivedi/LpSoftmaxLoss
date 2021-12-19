from src.normalization_layers import *
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import json


def train_resnet(dataset, logdir, num_epochs=50, order='l2', radius='ur', run_exp_count=1):
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

    def create_classifier(encoder, trainable=True):
        for layer in encoder.layers:
            layer.trainable = trainable
        inputs = keras.Input(shape=input_shape)
        features = encoder(inputs)

        if not (order == 'nl' and radius == 'nr'):
            normalization = get_normalization_layer(order, radius)
            features = normalization(features)

        outputs = layers.Dense(num_classes, activation="softmax")(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        return model

    def create_encoder():
        inputs = keras.Input(shape=input_shape)
        augmented = data_augmentation(inputs)
        resnet = keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape,
                                               pooling="avg")
        outputs = resnet(augmented)
        return keras.Model(inputs=inputs, outputs=outputs, name="resnet50-encoder")

    # repeat the exp for all counts
    for run_exp in range(run_exp_count):
        encoder = create_encoder()
        classifier = create_classifier(encoder)
        train_history = classifier.fit(train_ds, batch_size=batch_size, epochs=num_epochs,
                                       validation_data=validation_ds)

        # log training history
        json.dump(train_history.history, open(f"{logdir}/th_rn5_{order}_{radius}_run{run_exp}.json", 'w'), indent=4)
