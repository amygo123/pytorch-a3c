import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.NGPU1 = nn.GRUCell
        self.nnnn = nn.GRU
        self.Linear1 = nn.Linear(100, 72)
        self.Linear2 = nn.Linear(72, 36)
        self.Linear1.weight.data = normalized_columns_initializer(
            self.Linear1.weight.data, 0.01)
        self.Linear1.bias.data.fill_(0)
        self.Linear2.weight.data = normalized_columns_initializer(
            self.Linear2.weight.data, 0.01)
        self.Linear2.bias.data.fill_(0)

        num_outputs = action_space
        self.critic_linear = nn.Linear(36, 1)
        self.actor_linear = nn.Linear(36, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):

        x = self.Linear1(inputs)
        x = self.Linear2(x)
        return self.critic_linear(x), self.actor_linear(x)
