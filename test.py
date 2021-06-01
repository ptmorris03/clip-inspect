from pathlib import Path

import torch.nn as nn

from clip_inspect import get_grid, load_model, get_device, CLIPVisualResblockFF


model_path = Path("/data/clip/models/ViT-B-32.pt")
model = load_model(model_path)

block = CLIPVisualResblockFF(model, 0)


def test():
    grid = get_grid(100)

    dims = [0, 1]
    tensors = block(grid, in_dims=dims, res=False)
    for name, tensor in tensors.items():
        print(name, tensor.shape)

def test_gpu():
    device = get_device()
    block = block.to(device)

    grid = get_grid(100)
    grid = grid.to(device)

    dims = [0, 1]
    tensors = block(grid, in_dims=dims, res=False)
    for name, tensor in tensors.items():
        print(name, tensor.shape, tensor.device)

def test_gpu_parallel():
    device = get_device()
    
    block = nn.DataParallel(block)
    block = block.to(device)

    grid = get_grid(100)
    grid = grid.to(device)

    dims = [0, 1]
    tensors = block(grid, in_dims=dims, res=False)
    for name, tensor in tensors.items():
        print(name, tensor.shape, tensor.device)

def test_gpu_grad():
    device = get_device()

    block = nn.DataParallel(block)
    block = block.to(device)

    grid = get_grid(300)
    grid = grid.to(device)

    dims = [0, 1]
    out_grids = [grid.cpu()]
    for i in range(1000):
        tensors = block(
            grid, 
            in_dims=dims, 
            res=False, 
            return_keys=["out_project"]
        )
        grid = tensors["out_project"]
        out_grids.append(grid.cpu())
        print(len(out_grids))

if __name__ == "__main__":
    #test()
    #test_gpu()
    #test_gpu_parallel()
    test_gpu_grad()
