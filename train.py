import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import sys

directory = "submission"
user_data = directory + "/train"
valid_data = directory + "/val"
test_data = "label_book" # this can be the label book, or any other test set you create
n_epoch = 100

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8
tf.random.set_seed(123)


if __name__ == "__main__":
    train = tf.keras.preprocessing.image_dataset_from_directory(
        user_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        valid_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    total_length = ((train.cardinality() + valid.cardinality()) * batch_size).numpy()
    if total_length > 10_000:
        print(f"Dataset size larger than 10,000. Got {total_length} examples")
        sys.exit()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    loss_0, acc_0 = model.evaluate(valid)
    print(f"loss {loss_0}, acc {acc_0}")

    ## (disabled for local run) train model
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     "ckpts/best_model",
    #     monitor="val_accuracy",
    #     mode="max",
    #     save_best_only=True,
    #     save_weights_only=True,
    # )

    # history = model.fit(
    #     train,
    #     validation_data=valid,
    #     epochs=100,
    #     callbacks=[checkpoint],
    # )

    # model.load_weights("ckpts/best_model")

    ## load the weight file directly
    model.load_weights("ckpts/best_model")
    loss, acc = model.evaluate(valid)
    test_loss, test_acc = model.evaluate(test)

    ## write result to file
    with open('metrics.txt', 'w') as f:
        f.write(f"final loss: {loss}\n")
        f.write(f"final acc: {acc}\n")
        f.write(f"test loss {test_loss}\n")
        f.write(f"test acc {test_acc}")