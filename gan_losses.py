import tensorflow as tf


def get_acgan_losses_cross_entropy(D_real_source_logits, D_fake_source_logits, D_real_class_logits, D_fake_class_logits, L_real, L_fake, G_sample, X):

    D_loss_real_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_source_logits,
                                                                                labels=tf.ones_like(D_real_source_logits)))
    D_loss_fake_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_source_logits,
                                                                                labels=tf.zeros_like(D_fake_source_logits)))
    D_loss_real_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_class_logits,
                                                                               labels=L_real))
    D_loss_fake_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_class_logits,
                                                                               labels=L_fake))
    G_loss_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_source_logits,
                                                                           labels=tf.ones_like(D_fake_source_logits)))
    G_loss_dist = 7.0 * tf.reduce_mean(tf.squared_difference(G_sample, X))

    D_loss = D_loss_real_source + D_loss_real_class + D_loss_fake_source + D_loss_fake_class

    G_loss = G_loss_source + G_loss_dist + D_loss_real_class + D_loss_fake_class

    return D_loss, G_loss, G_loss_source, G_loss_dist


def get_infogan_losses_cross_entropy(logits_real, logits_fake, code_logits_fake, C):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                         logits=logits_real))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                         logits=logits_fake))
    D_loss = D_loss_real + D_loss_fake

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),
                                                                    logits=logits_fake))
    Q_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=C,
                                                                    logits=code_logits_fake))

    return D_loss, G_loss, Q_loss