import pickle as pkl
from math import sqrt
from random import randint

import numpy as np
from keras.utils.np_utils import to_categorical
from scipy import interpolate

em_path = 'D:\\data\\noise_examples\\em' + "_250Hz.pkl"
ma_path = 'D:\\data\\noise_examples\\ma' + "_250Hz.pkl"
bw_path = 'D:\\data\\noise_examples\\bw' + "_250Hz.pkl"

with open(em_path, 'rb') as f:
    em = pkl.load(f)
with open(ma_path, 'rb') as f:
    ma = pkl.load(f)
with open(bw_path, 'rb') as f:
    bw = pkl.load(f)


def batch_generator(ecgs, size, batch_size, noise_type='ma', level=None):
    if not isinstance(ecgs, list):
        signals = [ecgs]
    else:
        signals = ecgs

    while True:
        x_batch, y_batch, l_batch = get_batch(signals, size, batch_size, noise_type, level)

        l_batch = to_categorical(l_batch, num_classes=5)

        x_batch = np.expand_dims(x_batch, 2)
        y_batch = np.expand_dims(y_batch, 2)
        l_batch = np.expand_dims(l_batch, 2)
        x_batch = np.expand_dims(x_batch, 3)
        y_batch = np.expand_dims(y_batch, 3)
        l_batch = np.expand_dims(l_batch, 3)

        yield x_batch, y_batch, l_batch


def get_batch(ecgs, size, batch_size, noise_type, level):
    if level is None:
        l_batch = np.random.randint(low=0, high=5, size=batch_size)
    else:
        l_batch = np.array([level] * batch_size)

    x_batch = np.zeros((batch_size, size))
    y_batch = np.zeros((batch_size, size))
    for i in range(batch_size):
        list_pos = randint(0, len(ecgs) - 1)
        (x_batch[i], y_batch[i]) = get_noised_signal(ecgs[list_pos], size, noise_type, l_batch[i])

    return x_batch, y_batch, l_batch


def get_noised_signal(signal, size, noise_type, level):
    string_pos = randint(0, signal.shape[0] - 1)
    signal_pos = randint(0, signal.shape[1] - size - 1)

    sample = signal[string_pos, signal_pos:signal_pos + size]
    noised_sample = _add_noise(sample, noise_type, level)

    return (sample, noised_sample)


def _add_noise(ecg, noise_type='em', level=1):
    size = len(ecg)

    if level != 0:
        noise_sample, snr = _get_noise_snr(noise_type, level)

        noise_start_position = randint(0, noise_sample.shape[0] - size - 1)
        noise_channel = randint(0, 1)
        noise_fragment = noise_sample[noise_start_position:noise_start_position + size, noise_channel]

        noise_power = np.sum(noise_fragment ** 2, axis=0)
        ecg_power = np.sum(ecg ** 2, axis=0)

        target_noise = ecg_power / (10 ** (snr / 10))

        noise_fragment = noise_fragment * sqrt(target_noise) / sqrt(np.mean(noise_power))

    else:
        noise_fragment = np.zeros_like(ecg)

    return ecg + noise_fragment


def _get_noise_snr(noise_type, level):
    if noise_type == 'bw':
        noise_sample = bw
        if level == 1:
            snr = 12
        elif level == 2:
            snr = 6
        elif level == 3:
            snr = 0
        elif level == 4:
            snr = -6
    elif noise_type == 'em':
        noise_sample = em
        if level == 1:
            snr = 6
        elif level == 2:
            snr = 0
        elif level == 3:
            snr = -6
        elif level == 4:
            snr = -12
    elif noise_type == 'ma':
        noise_sample = ma
        if level == 1:
            snr = 12
        elif level == 2:
            snr = 6
        elif level == 3:
            snr = 0
        elif level == 4:
            snr = -6

    return noise_sample, snr


def _resize_signal(noise, old_freq=360, new_freq=500):
    len_seconds = (noise.shape[0] // old_freq)
    noise = noise[:len_seconds * old_freq]
    x = np.arange(0, len_seconds, 1 / old_freq)

    tck = interpolate.splrep(x, noise, s=0)
    xnew = np.arange(0, len_seconds, 1 / new_freq)
    ynew = interpolate.splev(xnew, tck, der=0)

    return ynew


def _resize_noise(noise, name, old_freq=360, new_freq=500):
    len_seconds = (noise.shape[0] // old_freq)
    new_noise = np.zeros((len_seconds * new_freq, noise.shape[1]))
    for i in range(noise.shape[1]):
        new_noise[:, i] = _resize_signal(noise[:, i], old_freq, new_freq)

    with open(name + '.pkl', 'wb') as output:
        pkl.dump(new_noise, output)


if __name__ == "__main__":
    from dataset import load_holter
    import matplotlib.pyplot as plt

    x = load_holter(patient=0)[1134000:1134000 + 5000:2]
    # srt_positions_0 = [646000, 1134000, 1590000, 1699000, 1801500, 1829000, 1870000, 1884000]
    # end_positions_0 = [655000, 1140000, 1617000, 1715000, 1807000, 1834500, 1879000, 1910500]

    import BaselineWanderRemoval as bwr

    x_fixed = bwr.fix_baseline_wander(x, 500)
    x_fixed = np.array(x_fixed)
    print(x_fixed.shape)
    x_noised_1 = _add_noise(x_fixed, 'ma', 1)
    x_noised_2 = _add_noise(x_fixed, 'ma', 2)
    x_noised_3 = _add_noise(x_fixed, 'ma', 3)
    x_noised_4 = _add_noise(x_fixed, 'ma', 4)

    plt.figure()
    plt.plot(x_fixed[:], color='k')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='--', color='red', linewidth='0.7')
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.4')
    plt.tight_layout()

    plt.figure()
    plt.plot(x_noised_1[:], color='k')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='--', color='red', linewidth='0.7')
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.4')
    plt.tight_layout()

    plt.figure()
    plt.plot(x_noised_2[:], color='k')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='--', color='red', linewidth='0.7')
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.4')
    plt.tight_layout()

    plt.figure()
    plt.plot(x_noised_3[:], color='k')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='--', color='red', linewidth='0.7')
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.4')
    plt.tight_layout()

    plt.figure()
    plt.plot(x_noised_4[:], color='k')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='--', color='red', linewidth='0.7')
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.4')
    plt.tight_layout()

    plt.show()
