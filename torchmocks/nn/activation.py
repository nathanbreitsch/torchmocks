import torch


class ActivationMock:
    def __init__(self, activation):
        self.__class__ = type(
            activation.__class__.__name__, (self.__class__, activation.__class__), {}
        )
        self.__dict__ = activation.__dict__

    def forward(self, x):
        return x


activations = [
    torch.nn.modules.activation.Threshold,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.RReLU,
    torch.nn.modules.activation.Hardtanh,
    torch.nn.modules.activation.ReLU6,
    torch.nn.modules.activation.Sigmoid,
    torch.nn.modules.activation.Hardsigmoid,
    torch.nn.modules.activation.Tanh,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.activation.Mish,
    torch.nn.modules.activation.Hardswish,
    torch.nn.modules.activation.ELU,
    torch.nn.modules.activation.CELU,
    torch.nn.modules.activation.SELU,
    torch.nn.modules.activation.GLU,
    torch.nn.modules.activation.GELU,
    torch.nn.modules.activation.Hardshrink,
    torch.nn.modules.activation.LeakyReLU,
    torch.nn.modules.activation.LogSigmoid,
    torch.nn.modules.activation.Softplus,
    torch.nn.modules.activation.Softshrink,
    torch.nn.modules.activation.PReLU,
    torch.nn.modules.activation.Softsign,
    torch.nn.modules.activation.Tanhshrink,
    torch.nn.modules.activation.Softmin,
    torch.nn.modules.activation.Softmax,
    torch.nn.modules.activation.Softmax2d,
    torch.nn.modules.activation.LogSoftmax,
    # why is this considered activation?
    # torch.nn.modules.activation.MultiheadAttention,
]

mock_dict = {a: ActivationMock for a in activations}
