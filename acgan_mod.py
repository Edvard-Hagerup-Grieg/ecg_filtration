import tensorflow as tf
from keras import backend as K

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from generators import batch_generator, _get_noise_snr
from dataset import load_holter, load_dataset, load_mit
from visualization import show_images
from metrics import get_metrics

from gan_models import generator, discriminator
from gan_losses import get_acgan_losses_cross_entropy


def training(dataset_train_ecg, dataset_test_ecg, ecg_len, batch_size, noise_type='ma', level=None, Y=None):
    num_noise_lavels = 5

    X = tf.placeholder(tf.float32, shape=[None, ecg_len, 1, 1], name='X')
    Z = tf.placeholder(tf.float32, shape=[None, ecg_len, 1, 1], name='Z')
    L = tf.placeholder(tf.float32, shape=[None, num_noise_lavels, 1, 1], name='L')
    isTrain = tf.placeholder(dtype=tf.bool, name='isTrain')

    L_real = tf.concat([tf.ones([batch_size, 1, 1, 1]), tf.zeros([batch_size, num_noise_lavels, 1, 1])], 1)
    L_fake = tf.concat([tf.zeros([batch_size, 1, 1, 1]), L], 1)

    G_sample = generator(Z, L, isTrain)
    D_real_source_logits, D_real_class_logits = discriminator(X, num_noise_lavels, isTrain)
    D_fake_source_logits, D_fake_class_logits = discriminator(G_sample, num_noise_lavels, isTrain, reuse=True)

    D_loss, G_loss, G_loss_source, G_loss_dist = get_acgan_losses_cross_entropy(
        D_real_source_logits=D_real_source_logits,
        D_fake_source_logits=D_fake_source_logits,
        D_real_class_logits=D_real_class_logits,
        D_fake_class_logits=D_fake_class_logits,
        L_real=L_real,
        L_fake=L_fake,
        G_sample=G_sample,
        X=X)

    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    D_solver = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_solver = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=G_vars)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dataset_train_data_generator = batch_generator(ecgs=dataset_train_ecg,
                                                       size=ecg_len,
                                                       batch_size=batch_size,
                                                       noise_type=noise_type,
                                                       level=level)
        dataset_test_data_generator = batch_generator(ecgs=dataset_test_ecg,
                                                      size=ecg_len,
                                                      batch_size=batch_size,
                                                      noise_type=noise_type,
                                                      level=level)

        z_test_dataset_batch, zn_test_dataset_batch, zl_test_dataset_batch = next(dataset_test_data_generator)
        z_train_dataset_batch, zn_train_dataset_batch, zl_train_dataset_batch = next(dataset_train_data_generator)

        test_dataset_samples = []
        test_dataset_samples_mse = []
        test_dataset_samples_snr = []
        test_dataset_samples_snr_legend = []

        train_dataset_samples = []
        train_dataset_samples_mse = []
        train_dataset_samples_snr = []
        train_dataset_samples_snr_legend = []

        D_loss_values = []
        G_loss_values = []
        G_loss_source_values = []
        G_loss_dist_values = []
        number_subplots = 3

        train_epoches = 5000
        for it in range(train_epoches + 1):

            if it % (train_epoches / 100) == 0:
                dataset_test_samples = sess.run(G_sample,
                                                feed_dict={Z: zn_test_dataset_batch,
                                                           L: zl_test_dataset_batch,
                                                           isTrain: True})
                dataset_train_samples = sess.run(G_sample,
                                                 feed_dict={Z: zn_train_dataset_batch,
                                                            L: zl_train_dataset_batch,
                                                            isTrain: True})

                test_snr_dt = []
                test_snr_dt_l = []
                train_snr_dt = []
                train_snr_dt_l = []
                for i in range(number_subplots):
                    snr_dt_test_, _ = get_metrics(z_test_dataset_batch[i:i + 1, :, 0, 0],
                                                  dataset_test_samples[i:i + 1, :, 0, 0])
                    snr_dt_train_, _ = get_metrics(z_train_dataset_batch[i:i + 1, :, 0, 0],
                                                   dataset_train_samples[i:i + 1, :, 0, 0])

                    test_snr_dt.append(snr_dt_test_)
                    test_snr_dt_l.append("snr %.2f" % (snr_dt_test_))

                    train_snr_dt.append(snr_dt_train_)
                    train_snr_dt_l.append("snr %.2f" % (snr_dt_train_))

                mse_dt_test_ = sess.run(
                    tf.reduce_mean(tf.squared_difference(z_test_dataset_batch, dataset_test_samples)))
                mse_dt_train_ = sess.run(
                    tf.reduce_mean(tf.squared_difference(z_train_dataset_batch, dataset_train_samples)))

                test_dataset_samples_snr.append(test_snr_dt)
                test_dataset_samples_snr_legend.append(test_snr_dt_l)
                test_dataset_samples_mse.append(mse_dt_test_)
                test_dataset_samples.append(dataset_test_samples[:, :, 0, 0])

                train_dataset_samples_snr.append(train_snr_dt)
                train_dataset_samples_snr_legend.append(train_snr_dt_l)
                train_dataset_samples_mse.append(mse_dt_train_)
                train_dataset_samples.append(dataset_train_samples[:, :, 0, 0])

                if it % train_epoches == 0:
                    plt.clf()
                    plt.plot(dataset_test_samples[0, :, 0, 0])
                    plt.plot(z_test_dataset_batch[0, :, 0, 0])
                    plt.legend(['pred', 'gt'])
                    plt.title('snr = {:.2}, mse = {:.2}'.format(test_snr_dt[0], mse_dt_test_))
                    plt.show()

            x, z, l = next(dataset_train_data_generator)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, Z: z, L: l, isTrain: True})
            _, G_loss_curr, G_loss_source_curr, G_loss_dist_curr = sess.run(
                [G_solver, G_loss, G_loss_source, G_loss_dist], feed_dict={X: x, Z: z, L: l, isTrain: True})

            D_loss_values.append(D_loss_curr)
            G_loss_values.append(G_loss_curr)
            G_loss_source_values.append(G_loss_source_curr)
            G_loss_dist_values.append(G_loss_dist_curr)

            if it % 1 == 0:
                print('Iter.: {}'.format(it))
                print('Dis. loss: {:.4}'.format(D_loss_curr))
                print('Gen. loss: {:.4}'.format(G_loss_curr))
                print()

        ################# SAVE RES. ################

        samples_dt = np.array(test_dataset_samples)

        show_images(signals=samples_dt[:, :number_subplots, :],
                    filename='test_dataset_ecg_denoised_2048_generating(acgan_mod_ce5)({})'.format(train_epoches),
                    legend=test_dataset_samples_snr_legend)

        samples_dt = np.array(train_dataset_samples)

        show_images(signals=samples_dt[:, :number_subplots, :],
                    filename='train_dataset_ecg_denoised_2048_generating(acgan_mod_ce5)({})'.format(train_epoches),
                    legend=train_dataset_samples_snr_legend)

        ################# SAVE GT #################

        lbls_test = [np.argmax(zl_test_dataset_batch[i, :, 0, 0])
                     for i in range(number_subplots)]

        test_dataset_gt_snr = []
        for i in lbls_test:
            if i == 0:
                snr_dt = np.Inf
            else:
                _, snr_dt = _get_noise_snr(noise_type='ma', level=i)
            test_dataset_gt_snr.append(snr_dt)

        test_dataset_legend = list("snr %.2f" % (test_dataset_gt_snr[i]) for i in range(number_subplots))

        fig1 = show_images(zn_test_dataset_batch[:number_subplots, :, 0, 0], legend=test_dataset_legend)
        plt.savefig("test_dataset_ecg_denoised_2048_generating(ngt)")
        fig2 = show_images(z_test_dataset_batch[:number_subplots, :, 0, 0])
        plt.savefig("test_dataset_ecg_denoised_2048_generating(gt)")

        lbls_train = [np.argmax(zl_train_dataset_batch[i, :, 0, 0])
                      for i in range(number_subplots)]

        train_dataset_gt_snr = []
        for i in lbls_train:
            if i == 0:
                snr_dt = np.Inf
            else:
                _, snr_dt = _get_noise_snr(noise_type=noise_type, level=i)
            train_dataset_gt_snr.append(snr_dt)

        train_dataset_legend = list("snr %.2f" % (train_dataset_gt_snr[i]) for i in range(number_subplots))

        fig1 = show_images(zn_train_dataset_batch[:number_subplots, :, 0, 0], legend=train_dataset_legend)
        plt.savefig("train_dataset_ecg_denoised_2048_generating(ngt)")
        fig2 = show_images(z_train_dataset_batch[:number_subplots, :, 0, 0])
        plt.savefig("train_dataset_ecg_denoised_2048_generating(gt)")

        ################# PLOT LOSS ###############
        plt.close('all')

        plt.clf()
        plt.plot(D_loss_values, linewidth=0.8)
        plt.plot(G_loss_values, linewidth=0.8)
        plt.legend(["Discr. loss", "Gener. loss"])
        plt.savefig("LOSS")

        plt.clf()
        plt.plot(G_loss_source_values, linewidth=0.8)
        plt.plot(G_loss_dist_values, linewidth=0.8)
        plt.legend(["Source loss", "Distance loss"])
        plt.savefig("G_LOSS")

        plt.clf()
        plt.plot(train_dataset_samples_mse, linewidth=0.8)
        plt.plot(test_dataset_samples_mse, linewidth=0.8)
        plt.legend(["MSE train.", "MSE test"])
        plt.savefig("MSE")

        test_dataset_samples_snr = np.array(test_dataset_samples_snr)
        test_signal_num = []
        plt.clf()
        for i in range(number_subplots):
            plt.plot(test_dataset_samples_snr[:, i], linewidth=0.8)
            test_signal_num.append("SNR for signal %d (start " % (i + 1) + test_dataset_legend[i] + ")")
        plt.legend(test_signal_num)
        plt.title("TEST")
        plt.savefig("SNR (test)")

        train_dataset_samples_snr = np.array(train_dataset_samples_snr)
        train_signal_num = []
        plt.clf()
        for i in range(number_subplots):
            plt.plot(train_dataset_samples_snr[:, i], linewidth=0.8)
            train_signal_num.append("SNR for signal %d (start " % (i + 1) + train_dataset_legend[i] + ")")
        plt.legend(train_signal_num)
        plt.title("TRAIN.")
        plt.savefig("SNR (train)")

        save_path = saver.save(sess, "model_n0_0123")
        print("Model saved in path: %s" % save_path)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    sess = tf.Session(config=config)
    K.set_session(sess)

    xy = load_dataset()
    X = xy["x"][:, ::2, 0]
    Y = xy["y"]

    dataset_train_ecg, dataset_test_ecg = train_test_split(X, test_size=0.33, random_state=42)
    print(dataset_train_ecg.shape)
    print(dataset_test_ecg.shape)

    ecg_len = 1024
    batch_size = 128

    training(dataset_train_ecg=dataset_train_ecg,
             dataset_test_ecg=dataset_test_ecg,
             ecg_len=ecg_len,
             batch_size=batch_size,
             noise_type='ma',
             level=None,
             Y=Y)