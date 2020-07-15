import numpy as np
from generators import batch_generator
from dataset import load_holter
import matplotlib.pyplot as plt


def get_metrics(x, xm):

    snr = np.zeros((x.shape[0], 1), dtype=np.float32)
    rmse = np.zeros((x.shape[0], 1), dtype=np.float32)

    for i in range(x.shape[0]):

        sum_x = np.sum(x[i]**2, axis=0)
        sum_d = np.sum((xm[i] - x[i])**2, axis=0)

        snr[i] = 10. * np.log10(sum_x / sum_d)
        rmse[i] = np.sqrt(sum_d / x.shape[1])

    return np.sum(snr) / x.shape[0], np.sum(rmse) / x.shape[0]


if __name__ == "__main__":
    xh = load_holter(patient=0)

    srt_positions = [646000, 1134000, 1590000, 1699000, 1801500, 1829000, 1870000, 1884000]
    end_positions = [655000, 1140000, 1617000, 1715000, 1807000, 1834500, 1879000, 1910500]

    x_list = []
    for i in range(len(srt_positions)):
        x_list.append(np.expand_dims(xh[srt_positions[i]: end_positions[i]], axis=0))

    level = 3
    train_data_generator = batch_generator(x_list, 2048, 5)

    x_, z_, l_ = next(train_data_generator)

    snr_v, rmse_v = get_metrics(x_[:, :, 0, 0], z_[:, :, 0, 0])
    print("mean ", snr_v, rmse_v)

    for i in range(x_.shape[0]):
        plt.clf()
        plt.plot(x_[i, :, 0, 0], color='black', linewidth=0.8)
        plt.plot(z_[i, :, 0, 0], color='black', linewidth=0.8, alpha=0.3)
        plt.legend(['ecg', 'noised ecg'])
        plt.title('level {}'.format(np.argmax(l_[i, :, 0, 0]) + 1))
        plt.show()