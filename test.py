from pathlib import Path

import torch.nn as nn

from clip_inspect import get_grid, load_model, get_device, CLIPVisualResblockFF


def test():
    model_path = Path("/data/clip/models/ViT-B-32.pt")
    model = load_model(model_path)

    block = CLIPVisualResblockFF(model, 0)

    grid = get_grid(100)

    dims = [0, 1]
    tensors = block(grid, in_dims=dims, res=False)
    for name, tensor in tensors.items():
        print(name, tensor.shape)

def test_gpu():
    model_path = Path("/data/clip/models/ViT-B-32.pt")
    model = load_model(model_path)

    device = get_device()

    block = CLIPVisualResblockFF(model, 0)
    block = block.to(device)

    grid = get_grid(100)
    grid = grid.to(device)

    dims = [0, 1]
    tensors = block(grid, in_dims=dims, res=False)
    for name, tensor in tensors.items():
        print(name, tensor.shape, tensor.device)

def test_gpu_parallel():
    model_path = Path("/data/clip/models/ViT-B-32.pt")
    model = load_model(model_path)

    device = get_device()

    block = CLIPVisualResblockFF(model, 0)
    block = nn.DataParallel(block)
    block = block.to(device)

    grid = get_grid(100)
    grid = grid.to(device)

    dims = [0, 1]
    tensors = block(grid, in_dims=dims, res=False)
    for name, tensor in tensors.items():
        print(name, tensor.shape, tensor.device)

def test_gpu_grad():
    model_path = Path("/data/clip/models/ViT-B-32.pt")
    model = load_model(model_path)

    device = get_device()

    block = CLIPVisualResblockFF(model, 0)
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
