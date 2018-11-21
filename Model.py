import tensorflow as tf

# Learning rate
Learning_rate = 0.001
# Optimizer used by the model, 0 for SGD, 1 for Adam, 2 for RMSProp
optimizer_type = 1
# Mini-batch size
batch_size = 128
# Activation function used , 0 for tanh, 1 for relu, 2 for sigmoid
activation_type = 1
# regularizer used , 0 for None, 1 for L1, 2 for L2
regularizer_type = 0
# Number of max epochs for training
epochs = 300
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = False
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 5
# Small epsilon value for the BN transform
epsilon = 1e-3


def get_inputs():
    """
    Generate the tf.placeholder for the model input.
    :return:
    inputs: input of the model, tensor of shape [batch_size, image_size]
    targets: targets(true result) used for training the CNN, tensor of shape
    [batch_size, class_number]
    learning_rate: learning rate for the mini-batch training.
    """
    inputs = tf.placeholder(tf.float32, [None, 12, 12, 4], name="inputs")
    targets = tf.placeholder(tf.float32, [None, 2], name="targets")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, keep_prob


def construct_CNN(input, filter, kernel_size, name):
    activation_collection = {0: tf.nn.tanh,
                             1: tf.nn.relu,
                             2: tf.nn.sigmoid
                             }
    regularizer_collection = {0: None,
                              1: tf.contrib.layers.l1_regularizer,
                              2: tf.contrib.layers.l2_regularizer}
    return tf.layers.conv2d(inputs=input,
                            filters=filter,
                            kernel_size=kernel_size,
                            padding="same",
                            activation=activation_collection[activation_type],
                            kernel_initializer=tf.truncated_normal_initializer,
                            kernel_regularizer=regularizer_collection[regularizer_type],
                            name=name
                            )


def construct_pool(input, pool_size, strides=2):
    return tf.layers.max_pooling2d(inputs=input,
                                   pool_size=pool_size,
                                   strides=strides)


def construct_DNN(inputs, units, activation,
                  kernel_initializer=tf.truncated_normal_initializer,
                  bias_initializer=tf.constant_initializer(0)):
    activation_collection = {0: tf.nn.tanh,
                             1: tf.nn.relu,
                             2: tf.nn.sigmoid,
                             3: None
                             }
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_collection[activation],
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer)


def batch_norm_wrapper(inputs, is_training, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    ema = tf.train.ExponentialMovingAverage(decay)
    pop_mean, pop_var = tf.nn.moments(inputs,axes=[0,1,2])

    def mean_var_with_update():
        ema_apply_op = ema.apply([pop_mean, pop_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(pop_mean), tf.identity(pop_var)

    mean, var = tf.cond(tf.constant(is_training, dtype=tf.bool), mean_var_with_update,
                        lambda: (ema.average(pop_mean),ema.average(pop_var)))
    return tf.nn.batch_normalization(inputs,
                                     mean, var, beta, scale, epsilon)


def build_graph(isTrain):
    graph = tf.Graph()
    with graph.as_default():
        inputs, targets, keep_prob = get_inputs()
        conv1 = construct_CNN(inputs, 32, [3, 3], "convolution1")
        pool1 = construct_pool(conv1, [2, 2])
        pool1 = batch_norm_wrapper(pool1, isTrain)
        conv2 = construct_CNN(pool1, 64, [3, 3], "convolution2")
        pool2 = construct_pool(conv2, [2, 2])
        pool2 = batch_norm_wrapper(pool2, isTrain)
        conv3 = construct_CNN(pool2, 128, [3, 3], "convolution3")
        pool3 = construct_pool(conv3, [2, 2])
        pool3 = batch_norm_wrapper(pool3, isTrain)
        h_flat = tf.reshape(pool3, [-1, pool3.get_shape()[1] * pool3.get_shape()[2] * pool3.get_shape()[3]])
        h_fc1 = construct_DNN(h_flat, 128, 1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        y_ = construct_DNN(h_fc1_drop, 2, 3)
        prediction = tf.argmax(tf.nn.sigmoid(y_), 1,name='prediction')
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=y_))
        global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Learning_rate, global_steps, 1000, 0.5, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_steps)
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged = tf.summary.merge_all()
    return inputs, targets, keep_prob, prediction, train_step, merged, graph, accuracy, loss, learning_rate
