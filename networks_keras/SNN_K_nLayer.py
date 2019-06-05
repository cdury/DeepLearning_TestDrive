import os, sys
import time

import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import Input, Dense
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
)
from helper_nn.helper_load_data import mnist_data as data

# Logging
# 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# pathes
HOME_PATH = os.getcwd()
print(HOME_PATH)
TB_PATH = os.path.join(HOME_PATH, "tensorboard")
MODEL_PATH = os.path.join(HOME_PATH, "models")
DATA_PATH = os.path.join(HOME_PATH, "data")
TRAIN_DATA = "train/"
TEST_DATA = "test/"
# tensorboard log
TB_NAME = os.path.splitext(os.path.basename(__file__))[0]


def define_model(n_input, n_hidden_1, n_hidden_2, n_classes, learning_rate):
    # Start defining the input tensor:
    input = Input((n_input,))

    # create the layers and pass them the input tensor to get the output tensor:
    layer_1 = Dense(
        units=n_hidden_1,
        activation="relu",
        # kernel_regularizer=regularizers.l2(0.0002),
        # kernel_initializer="he_normal",
    )(input)

    layer_2 = Dense(
        units=n_hidden_2,
        activation="relu",
        # kernel_regularizer=regularizers.l2(0.0002),
        # kernel_initializer="he_normal",
    )(layer_1)
    out_layer = Dense(
        units=n_classes,
        # kernel_initializer="he_normal",
        # kernel_regularizer=regularizers.l2(0.0002),
        activation="softmax",
    )(layer_2)

    # define the model's start and end points
    model = Model(inputs=input, outputs=out_layer)

    loss_fn = lambda y_true, y_pred: tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=y_pred, labels=y_true
    )
    model.compile(
        # loss="categorical_crossentropy",
        loss=loss_fn,
        # optimizer=Adam(lr=learning_rate),
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model.summary()

    return model


def train_model(model, train_data, test_data, batch_size, training_epochs):
    # Data
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Callbacks
    # for Tensorboard evaluation
    tensorboard = TensorBoard(
        log_dir=os.path.join(TB_PATH, TB_NAME),
        update_freq="epoch",
        write_graph=True,
        write_images=True,
        histogram_freq=0,
        write_grads=False,
    )
    # for saving network with weights
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            MODEL_PATH, TB_NAME, "weights.{epoch:03d}-{val_loss:.2f}.hdf5"
        ),
        monitor="val_loss",  # monitor="val_acc",
        save_best_only=True,
        mode="min",  # mode="max",
        verbose=1,
        period=1,
    )
    if not os.path.isdir(os.path.join(MODEL_PATH, TB_NAME)):
        os.mkdir(os.path.join(MODEL_PATH, TB_NAME))
    # for early termination
    earlyterm = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [tensorboard, checkpoint, earlyterm]

    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=training_epochs,
        verbose=2,
        callbacks=callbacks,
    )

    print("Training Finished!")
    return history


def main():
    # Startup Parameters

    # Data
    # # Loading
    # Import MINST data
    X_train, y_train, X_test, y_test, mnist = data()
    # # Features
    train_data = X_train, y_train
    test_data = X_test, y_test

    # Graph
    # # Parameter
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    display_step = 1
    # # Network Parameters
    n_hidden_1 = 256  # 1st layer number of features
    n_hidden_2 = 256  # 2nd layer number of features
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)

    print("Measure Model SetUp")
    start_process_time = time.process_time()
    start_perf_time = time.perf_counter()
    model = define_model(n_input, n_hidden_1, n_hidden_2, n_classes, learning_rate)
    print(f"{time.process_time()-start_process_time}s with time.process_time()")
    print(f"{time.perf_counter()-start_perf_time}s with time.perf_counter()")

    # # Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    # Launch the  Training
    print("Measure Model Training")
    start_process_time = time.process_time()
    start_perf_time = time.perf_counter()
    train_model(model, train_data, test_data, batch_size, training_epochs)
    print(f"{time.process_time() - start_process_time}s with time.process_time()")
    print(f"{time.perf_counter() - start_perf_time}s with time.perf_counter()")

    # Test model
    score = model.evaluate(X_test, y_test, verbose=0)
    # Calculate accuracy
    print("Accuracy:", score)


if __name__ == "__main__":
    main()
