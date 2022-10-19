import torch
from torchmocks.torchmocks import mock_with_fx_graph_manipulation


class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.ones(3, 3)
        self.bias = torch.ones(1, 3)
        self.linear2 = torch.nn.Linear(3, 8)

    def forward(self, x):
        x1 = torch.nn.functional.linear(x, self.weight, self.bias)
        return self.linear2(x1)


def test_mocK_with_fx_graph_manipulation():
    net = Example()
    net = mock_with_fx_graph_manipulation(net)
    out = net(torch.ones((1, 3)))
    assert out.sum() == 0
