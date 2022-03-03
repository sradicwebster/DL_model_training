from typing import List, Type
import torch.nn as nn


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    softmax_output: bool = False,
) -> List[nn.Module]:

    assert len(net_arch) > 0
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    if softmax_output:
        modules.append(nn.Softmax(dim=1))

    return modules


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 net_arch: List[int],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 squash_output: bool = False,
                 softmax_output: bool = False,
                 ):
        super(MLP, self).__init__()
        modules = create_mlp(input_dim, output_dim, net_arch, activation_fn, squash_output,
                             softmax_output)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
