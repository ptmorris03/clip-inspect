from pathlib import Path
import click
import sys

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

sys.path.append("/code/")
from clip_inspect import get_grid, load_model, get_device, CLIPVisualResblockFF

def get_resblock(path, layer_idx):
    model_path = Path(path)
    model = load_model(model_path)

    device = get_device()

    block = CLIPVisualResblockFF(model, 0)
    block = nn.DataParallel(block)
    block = block.to(device)

    return block, device

def get_angle_grid(n_steps):
    angles = get_grid(n_steps, dims=1, sizes=(-np.pi, np.pi))[:,0]
    grid = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    
    return angles, grid

def get_angles(block, angles, grid, pc_dims, n_iters):
    angles = [angles]
    for i in range(n_iters):
        tensors = block(
            grid, 
            in_dims=pc_dims, 
            res=False, 
            return_keys=["out_project"]
        )
        grid = tensors["out_project"]
        out_angles = torch.atan2(grid[...,1], grid[...,0])
        angles.append(out_angles.cpu())
    
    return torch.stack(angles)

def plot_fractal(angles, out_path):
    n = angles.shape[0]
    r = np.linspace(1, n, num=n)
    th = np.linspace(-np.pi, np.pi, num=angles.shape[1])

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection' :'polar'}, figsize=(20, 20))
    ax.set_rlim(0.9, n)
    ax.pcolormesh(th, r, angles.numpy())
    ax.get_yaxis().set_visible(False)

    ax.tick_params(axis='x', which='both', bottom=False, top=True, colors='white', length=9, width=3, labelsize=20, pad=12)
    ax.set_facecolor('black')
    ax.set_xticklabels(["0", "π/4", "π/2", "3π/4", "π", "-3π/4", "-π/2", "-π/4"])
    fig.patch.set_facecolor('black')
    
    fig.savefig(out_path, dpi=108, facecolor=fig.get_facecolor(), pad_inches=0)
    
    #u know when u lock the car 6 times, just to make sure?
    plt.clf()
    plt.cla()
    ax.clear()
    plt.close()


@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--layer', type=int, required=True)
@click.option('--save_dir', type=click.Path(), required=True)
@click.option('--n_steps', type=int, default=100000)
@click.option('--n_iters', type=int, default=100)
def plot_all_pc_pairs(model_path, layer, save_dir, n_steps, n_iters):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    block, device = get_resblock(Path(model_path), layer)
    start_angles, grid = get_angle_grid(n_steps)
    grid = grid.to(device)

    dims = [0, 1]
    for i in range(768):
        for j in range(768):
            dims = [i, j]
            angles = get_angles(block, start_angles, grid, dims, n_iters)
            
            out_name = F"fractal_layer{layer+1:02d}_PC{dims[0]+1:03d}_PC{dims[1]+1:03d}.png"
            out_path = Path(save_dir, out_name)

            plot_fractal(angles, out_path)

            print(dims)

if __name__ == "__main__":
    plot_all_pc_pairs()
