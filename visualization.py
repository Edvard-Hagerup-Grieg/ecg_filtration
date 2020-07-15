import matplotlib.pyplot as plt
from celluloid import Camera


def update_axes(axes, signals, legend, handlelength):
    for i in range(signals.shape[0]):
        axes[i].plot(signals[i, :], color='black', linewidth=0.8)

        if legend is not None:
            axes[i].legend([legend[i]], loc=1, handlelength=handlelength)


def show_images(signals, filename='unnamed', **args):

    legend, handlelength = unpack(args)

    fig, axes = plt.subplots(signals.shape[signals.ndim % 2])

    if signals.ndim == 2:
        update_axes(axes, signals, legend, handlelength)
        plt.tight_layout()

    elif signals.ndim == 3:
        camera = Camera(fig)
        for i in range(signals.shape[0]):
            update_axes(axes, signals[i,:,:], legend[i], handlelength)
            camera.snap()

        plt.tight_layout()
        animation = camera.animate(repeat=False)
        animation.save(filename + '.gif', writer='imagemagick')

    return fig


def unpack(args):
    legend = None
    handlelength = None
    for name, value in args.items():
        if name == 'legend': legend = value
        elif name == 'handlelength': handlelength = value

    return legend, handlelength


if __name__ == "__main__":

    import numpy as np
    from dataset import load_holter

    len = 2048
    xh_0 = load_holter(patient=0)[646000:646000 + len]
    xh_3 = load_holter(patient=3)[69000+200:69000+200 + len]
    xh_6 = load_holter(patient=6)[549500:549500 + len]

    x = np.array([xh_0, xh_3, xh_6])
    fig = show_images(x, legend=['patient 1', 'patient 2', 'patient 3'])
    plt.savefig('holter_patients_example.png')