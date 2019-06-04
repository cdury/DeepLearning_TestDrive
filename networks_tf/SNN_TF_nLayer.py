import os, sys


import tensorflow as tf

# pathes
HOME_PATH = os.getcwd()
TB_PATH = os.path.join(HOME_PATH, "tensorboard")
DATA_PATH = os.path.join(HOME_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, "MNIST")
TRAIN_DATA = "train/"
TEST_DATA = "test/"
# tensorboard log
TB_NAME = os.path.splitext(os.path.basename(__file__))[0]


def data():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets(DATASET_PATH, one_hot=True)
    X_train, y_train = mnist.train.next_batch(mnist.train.num_examples)
    X_test, y_test = mnist.test.next_batch(mnist.test.num_examples)
    return X_train, y_train, X_test, y_test, mnist


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
    print("Optimization Finished!")


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

    model, placeholder = define_graph(
        n_input, n_hidden_1, n_hidden_2, n_classes, learning_rate
    )
    pred, optimizer, cost = model
    x, y = placeholder

    # # Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    # Launch the graph & Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_graph_weights(
            model, placeholder, sess, mnist, batch_size, training_epochs, display_step
        )

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


if __name__ == "__main__":
    main()
