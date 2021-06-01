from clip_inspect.model import CLIP
from typing import Iterable, List, Optional

import torch
from torch import nn

from sklearn.decomposition import PCA


def get_grid(n: int = 300, dims: int = 2, sizes: Iterable = (-1.0, 1.0)):
    if type(sizes[0]) in [float, int]:
        sizes = [sizes for i in range(dims)]
        
    steps = [torch.linspace(sizes[i][0], sizes[i][1], n) for i in range(dims)]
    return torch.stack(torch.meshgrid(*steps), axis=-1)


class Components(nn.Module):
    def __init__(self, tensor, n_dims: int = None):
        super(Components, self).__init__()

        tensor = tensor.view(-1, tensor.shape[-1])
        tensor_np = tensor.detach().cpu().numpy()

        self.pca = PCA(n_dims).fit(tensor_np)

        components = torch.tensor(
            self.pca.components_,
            dtype=torch.float
        )
        self.register_buffer("components", components)
        self.n_dims = components.shape[1]
        
        variance = torch.tensor(
            self.pca.explained_variance_ratio_,
            dtype=torch.float
        )
        self.register_buffer("variance", variance)

    def forward(self, tensor, dims: List[int] = None, reverse: bool = False):
        if reverse:
            return self._to_grid(tensor, dims)
        else:
            return self._from_grid(tensor, dims)
    
    def _from_grid(self, grid, dims=None):
        if dims == None:
            dims = range(grid.shape[-1])

        return torch.matmul(grid, self.components[dims])

    def _to_grid(self, emb, dims=None):
        if dims == None:
            dims = range(self.n_dims)

        return torch.matmul(emb, self.components[dims].T)


class CLIPVisualResblockFF(nn.Module):
    def __init__(self, model, block_idx: int):
        super(CLIPVisualResblockFF, self).__init__()

        self.block = model.visual.transformer.resblocks[block_idx]
        self.components = Components(self.block.mlp.c_fc.weight)

    def forward(
        self,
        input,
        in_project: bool = True,
        in_dims: Optional[Iterable] = None,
        in_norm: bool = True,
        mlp: bool = True,
        res: bool = True,
        out_norm: bool = False,
        out_project: bool = True,
        out_dims: Optional[Iterable] = None,
        return_keys: Optional[Iterable[str]] = None
        ):
        if out_dims is None:
            out_dims = in_dims

        def store(dict, key, tensor, return_keys):
            if return_keys is None or key in return_keys:
                dict[key] = tensor

        tensors = {}

        if in_project:
            input = self.components(input, in_dims)
            store(tensors, "in_project", input, return_keys)

        if in_norm:
            out = self.block.ln_2(input)
            store(tensors, "in_norm", out, return_keys)

        if mlp:
            out = self.block.mlp(out)
            store(tensors, "mlp", out, return_keys)

        if res:
            out = out + input
            store(tensors, "res", out, return_keys)

        if out_norm:
            out = self.block.ln_2(out)
            store(tensors, "out_norm", out, return_keys)

        if out_project:
            out = self.components(out, out_dims, reverse=True)
            store(tensors, "out_project", out, return_keys)

        return tensors
