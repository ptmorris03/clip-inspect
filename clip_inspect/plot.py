import torch
import numpy as np

import matplotlib.pyplot as plt


def plot_norms(norms, cols=4, inches_per_fig=8):
    rows = norms.shape[0] // cols
    rows += 1 if norms.shape[0] % cols != 0 else 0

    figsize = (cols * inches_per_fig, rows * inches_per_fig)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(norms.shape[0]):
        row_idx = i // cols
        col_idx = i % cols
        axis = ax[row_idx][col_idx]

        axis.imshow(norms[i])
        axis.axis('off')
        axis.title.set_text(F"{i+1}")

    plt.tight_layout()
    plt.show()

def plot_grids(grids, cols=4, inches_per_fig=8, dot_size=0.01):
    rows = grids.shape[0] // cols
    rows += 1 if grids.shape[0] % cols != 0 else 0

    num_dots = int(torch.numel(grids) / (grids.shape[0] * grids.shape[-1]))
    colors=np.linspace(0, 1, num_dots)

    figsize = (cols * inches_per_fig, rows * inches_per_fig)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(grids.shape[0]):
        row_idx = i // cols
        col_idx = i % cols
        axis = ax[row_idx][col_idx]

        x = grids[i,...,0].view(-1)
        y = grids[i,...,1].view(-1)

        axis.scatter(x, y, c=colors, s=dot_size)
        axis.set_facecolor("black")
        #axis.axis('off')
        axis.title.set_text(F"{i}")

    plt.tight_layout()
    plt.show()
