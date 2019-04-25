import torch
import torch.nn as nn


class CGRUCell(nn.Module):
    def __init__(self, channels=4, k_size=3):
        super(CGRUCell, self).__init__()
        m = channels
        self.u = nn.Sequential(nn.Conv2d(m, m, k_size, padding=1), torch.nn.Dropout(0.09), nn.Sigmoid())
        self.r = nn.Sequential(nn.Conv2d(m, m, k_size, padding=1), torch.nn.Dropout(0.08), nn.Sigmoid())
        self.conv1 = nn.Conv2d(m, m, k_size, padding=1)

    def forward(self, s):
        u = torch.clamp(1.2 * self.u(s) - 0.1, min=0., max=1.)
        r = torch.clamp(1.2 * self.r(s) - 0.1, min=0., max=1.)
        return u * s + (1. - u) * torch.tanh(self.conv1(r * s))


class NeuralGPU(nn.Module):
    def __init__(self, ori_i, m, m_image_width, hidden_channel=64, kernel_size=3):
        super(NeuralGPU, self).__init__()
        self.m = m
        self.ori_i = ori_i
        self.emb = nn.Conv2d(ori_i, hidden_channel, kernel_size, padding=1)

        self.cgru1 = CGRUCell(channels=hidden_channel, k_size=kernel_size)
        self.cgru2 = CGRUCell(channels=hidden_channel, k_size=kernel_size)
        self.width = m_image_width
        self.last_layer = nn.Conv2d(64, 5, kernel_size, padding=1)

    def step(self, s):
        _s = self.cgru1(s)
        return self.cgru2(_s)

    def forward(self, inputs):
        m_image = self.make_mental_image(inputs).unsqueeze(0)#.cuda()
        embedded_image = self.emb(m_image)
        s = embedded_image
        for _ in range(inputs.shape[0]):
            s = self.step(s)
        out = self.last_layer(s)
        return out[:, :, :, 0]

    def make_mental_image(self, inputs):
        mental_image = torch.zeros((inputs.shape[0], self.width, self.ori_i))
        mental_image[:, 0, :] = inputs
        return mental_image.permute(2, 0, 1)


if __name__ == "__main__":
    ngpu = NeuralGPU(20, 5, 4)#.cuda()
    os = torch.zeros((5, 20))#.cuda()
    s = ngpu(os)[0]
    ss = s.permute(1, 0)
    print(s)
    print(ss)
    print(1)

