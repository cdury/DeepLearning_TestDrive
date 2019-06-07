import os, sys
import time

import tensorflow as tf
from helper_nn.helper_load_data import mnist_data as data

# Logging
# 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# pathes
HOME_PATH = os.getcwd()
print(HOME_PATH)
TB_PATH = os.path.join(HOME_PATH, "tensorboard")
DATA_PATH = os.path.join(HOME_PATH, "data")
TRAIN_DATA = "train/"
TEST_DATA = "test/"
# tensorboard log
TB_NAME = os.path.splitext(os.path.basename(__file__))[0]


def define_graph(n_input, n_hidden_1, n_hidden_2, n_classes, learning_rate):
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights["out"]) + biases["out"]
        return out_layer

    # Store layers weight & bias
    weights = {
        "h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        "h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        "out": tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([n_hidden_1])),
        "b2": tf.Variable(tf.random_normal([n_hidden_2])),
        "out": tf.Variable(tf.random_normal([n_classes])),
    }

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    placeholder = (x, y)

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    model = pred, optimizer, cost
    return model, placeholder


def train_graph_weights(
    model, placeholder, sess, mnist, batch_size, training_epochs, display_step
):
    pred, optimizer, cost = model
    x, y = placeholder
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Training Finished!")


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
    model, placeholder = define_graph(
        n_input, n_hidden_1, n_hidden_2, n_classes, learning_rate
    )
    pred, optimizer, cost = model
    x, y = placeholder
    print(f"{time.process_time()-start_process_time}s with time.process_time()")
    print(f"{time.perf_counter()-start_perf_time}s with time.perf_counter()")
    # # Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    # Launch the graph & Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("Measure Model Training")
        start_process_time = time.process_time()
        start_perf_time = time.perf_counter()
        train_graph_weights(
            model, placeholder, sess, mnist, batch_size, training_epochs, display_step
        )
        print(f"{time.process_time() - start_process_time}s with time.process_time()")
        print(f"{time.perf_counter() - start_perf_time}s with time.perf_counter()")
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


if __name__ == "__main__":
    main()
