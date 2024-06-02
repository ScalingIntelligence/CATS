import torch.nn as nn
from transformers import MistralModel

def compress_mistral(model: MistralModel):
    assert isinstance(model, MistralModel), f"{type(model)} is not MistralModel."
    model.layers = model.layers[:2]



