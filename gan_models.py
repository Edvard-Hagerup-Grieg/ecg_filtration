import tensorflow as tf

from activations import lrelu



def generator(z, l, isTrain=True, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):

        init = tf.contrib.layers.xavier_initializer()

        conv1 = tf.layers.conv2d(z, 64, [5, 1], strides=(1, 1), padding='same', kernel_initializer=init)
        lrelu1 = lrelu(conv1, 0.2)
        conv1_output = tf.layers.max_pooling2d(lrelu1, (2, 1), (2, 1))


        conv2 = tf.layers.conv2d(conv1_output, 128, [8, 1], strides=(1, 1), padding='same', kernel_initializer=init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        #lrelu2 = lrelu(conv2, 0.2)
        conv2_output = tf.layers.max_pooling2d(lrelu2, (2, 1), (2, 1))

        conv3 = tf.layers.conv2d(conv2_output, 256, [8, 1], strides=(1, 1), padding='same', kernel_initializer=init)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        #lrelu3 = lrelu(conv3, 0.2)
        conv3_output = tf.layers.max_pooling2d(lrelu3, (2, 1), (2, 1))

        conv4 = tf.layers.conv2d(conv3_output, 512, [8, 1], strides=(1, 1), padding='same', kernel_initializer=init)
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        #lrelu3 = lrelu(conv3, 0.2)
        conv4_output = tf.layers.max_pooling2d(lrelu4, (2, 1), (2, 1))


        conv5 = tf.layers.conv2d(conv4_output, 512, [8, 1], strides=(1, 1), padding='same', kernel_initializer=init)
        lrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)


        conv6_input = tf.concat([conv4_output, lrelu5], -1)
        conv6 = tf.layers.conv2d_transpose(conv6_input, 512, [8, 1], strides=(2, 1), padding='same', kernel_initializer=init)
        lrelu6 = lrelu(tf.layers.batch_normalization(conv6, training=isTrain), 0.2)
        #lrelu4 = lrelu(conv4, 0.2)

        conv7_input = tf.concat([conv3_output, lrelu6], -1)
        conv7 = tf.layers.conv2d_transpose(conv7_input, 256, [8, 1], strides=(2, 1), padding='same', kernel_initializer=init)
        lrelu7 = lrelu(tf.layers.batch_normalization(conv7, training=isTrain), 0.2)
        #lrelu5 = lrelu(conv5, 0.2)

        conv8_input = tf.concat([conv2_output, lrelu7], -1)
        conv8 = tf.layers.conv2d_transpose(conv8_input, 128, [8, 1], strides=(2, 1), padding='same', kernel_initializer=init)
        lrelu8 = lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)
        #lrelu5 = lrelu(conv5, 0.2)

        conv9 = tf.layers.conv2d_transpose(lrelu8, 1, [4, 1], strides=(2, 1), padding='same', kernel_initializer=init)
        outputs = tf.nn.tanh(conv9)

    return outputs


def discriminator(x, num_classes, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):

        init = tf.contrib.layers.xavier_initializer()

        conv1 = tf.layers.conv2d(x, 128, [4, 1], strides=(2, 1), padding='same', kernel_initializer=init)
        lrelu1 = lrelu(conv1, 0.2)

        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 1], strides=(2, 1), padding='same', kernel_initializer=init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        #lrelu2 = lrelu(conv2, 0.2)

        conv3 = tf.layers.conv2d(lrelu2, 256, [4, 1], strides=(4, 1), padding='same', kernel_initializer=init)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        #lrelu3 = lrelu(conv3, 0.2)

        conv4 = tf.layers.conv2d(lrelu3, 512, [4, 1], strides=(4, 1), padding='same', kernel_initializer=init)
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        #lrelu4 = lrelu(conv4, 0.2)

        conv5 = tf.layers.conv2d(lrelu4, 1024, [4, 1], strides=(4, 1), padding='same', kernel_initializer=init)
        lrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)
        #lrelu5 = lrelu(conv5, 0.2)

        flatten_lrelu5 = tf.layers.flatten(lrelu5)

        dense1_1 = tf.layers.dense(flatten_lrelu5, 1, kernel_initializer=init)
        source_logits = tf.reshape(dense1_1, [-1, 1, 1, 1])


        dense1_2 = tf.layers.dense(flatten_lrelu5, num_classes + 1, kernel_initializer=init)
        class_logits = tf.reshape(dense1_2, [-1, num_classes + 1, 1, 1])

        return source_logits, class_logits