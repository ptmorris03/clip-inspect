import clip_inspect.weights as clip_weights
import clip_inspect.model as clip_model
import numpy as np
from functools import partial

print(dir(clip_weights))

state_dict = clip_weights.load("ViT-B-32")
mlp = clip_model.MLP(state_dict, "transformer.resblocks.0")
print("hi here")
#print(mlp.params)
print(mlp.forward(np.zeros(512)))

#state_dict = clip_weights.load("ViT-B-16")
