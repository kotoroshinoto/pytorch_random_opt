import torch
from torch import nn
from collections import OrderedDict


class ANNClassifier(nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int = 2,
            activation_func=nn.ReLU,
            num_hidden_layers=1,
            size_hidden_layers=512,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        layers = OrderedDict()
        topology = [n_inputs] + ([size_hidden_layers] * num_hidden_layers) + [n_outputs]
        activ_layers_added= 0
        linear_layers_added = 0
        for i in range(len(topology)-1):
            if i > 0:
                layers[f"{activation_func.__name__}_{activ_layers_added}"] = activation_func()
                activ_layers_added += 1
            in_size = topology[i]
            out_size = topology[i+1]
            layers[f"{nn.Linear.__name__}_{linear_layers_added}"] = nn.Linear(in_size, out_size)
            linear_layers_added += 1
        layers['softmax'] = nn.Softmax(dim=0)
        self.linear_stack = nn.Sequential(layers)

    def forward(self, x):
        logits = self.linear_stack(x.to(torch.float32))
        return logits
