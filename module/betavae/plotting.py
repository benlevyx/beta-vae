import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def display_images(images, labels=None, ncols=4, nrows=None):
    num_images = len(images)
    nrows = nrows or int(np.ceil(num_images / ncols))
    fig, axs = plt.subplots(nrows, ncols, dpi=150)
    labels = labels if labels is not None else list(range(num_images))
    for ax, im, lab in zip(axs.flat, images, labels):
        display_image_single(im, label=lab, ax=ax)
    for i in range(num_images, axs.size):
        fig.delaxes(axs.flat[i])
    fig.subplots_adjust(wspace=0.6)


def display_image_single(image, label=None, ax=None):
    ax = ax or plt.gca()
    ax.imshow(image.squeeze(0).numpy().transpose(), cmap='gray')
    if label:
        ax.set_title(label)
    ax.set_axis_off()


def display_preds_trg(preds, trg, labels=None):
    num_pairs = len(preds)
    ncols = 3
    nrows = num_pairs
    fig, axs = plt.subplots(nrows, ncols, dpi=150)
    for i in range(num_pairs):
        lab = None if labels is None else labels[i]
        axs[i, 0].set_axis_off()
        axs[i, 0].annotate(lab, (0, 0.5))
        for j, tnsr in enumerate([preds, trg]):
            display_image_single(tnsr[i], label=lab, ax=axs[i, j+1])


def traverse_latent_space(model, dim1, dim2, ndims,
                          dim1_extent=(-4,4), dim2_extent=(-4,4),
                          nrows=10, ncols=10, ax=None):
    ax = ax or plt.gca()
    n_samples = nrows * ncols
    samples = np.zeros((n_samples, ndims))
    idx = 0
    xs, ys = [], []
    for i in np.linspace(-4, 4, ncols):
        for j in np.linspace(-4, 4, nrows):
            samples[idx, dim2] = i
            samples[idx, dim1] = j
            idx += 1
            xs.append(i)
    samples = torch.tensor(samples).float()
    with torch.no_grad():
        x = model.decode(samples)
    grid = np.zeros((28 * nrows, 28 * ncols))
    for idx in range(x.size(0)):
        row = idx // ncols
        col = idx % nrows
        grid[row * 28:(row + 1) * 28, col * 28:(col+1) * 28] = x[idx].transpose(1, 2).numpy()
    ax.imshow(grid, cmap='gray')
    ax.set_axis_off()


def plot_latent_space(model, loader, ax=None, label_dict=None):
    ax = ax or plt.gca()
    model.eval()
    embeddings = []
    labels = []
    for inputs, labs in loader:
        with torch.no_grad():
            z = model.embed(inputs)
        embeddings.append(z)
        labels.append(labs)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    for cls in set(labels.tolist()):
        idxs = labels == cls
        if label_dict is not None:
            cls = label_dict[cls]
        ax.scatter(embeddings[idxs, 0],
                    embeddings[idxs, 1],
                    label=cls,
                    alpha=0.5,
                    s=6)
    ax.legend()