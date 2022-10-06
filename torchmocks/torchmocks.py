import torch


class Conv2dMock:
    def __init__(self, conv2d):
        self.__class__ = type(
            conv2d.__class__.__name__, (self.__class__, conv2d.__class__), {}
        )
        self.__dict__ = conv2d.__dict__
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.dilation = conv2d.dilation
        self.device = conv2d.weight.device
        self.mock_gradient_sink = torch.ones(1, requires_grad=True)

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        assert self.in_channels == in_channels
        assert self.device == x.device
        out_height = (
            in_height
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
            + self.stride[0]
        ) // self.stride[0]
        out_width = (
            in_width
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
            + self.stride[1]
        ) // self.stride[1]

        return self.mock_gradient_sink * torch.zeros(
            (batch_size, self.out_channels, out_height, out_width)
        )


class Norm2dMock:
    def __init__(self, batch_norm):
        self.__class__ = type(
            batch_norm.__class__.__name__, (self.__class__, batch_norm.__class__), {}
        )
        self.__dict__ = batch_norm.__dict__
        self.num_features = batch_norm.num_features

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        assert self.num_features == in_channels
        return x


class LinearMock:
    def __init__(self, linear):
        self.__class__ = type(
            linear.__class__.__name__, (self.__class__, linear.__class__), {}
        )
        self.__dict__ = linear.__dict__
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.mock_gradient_sink = torch.ones(1, requires_grad=True)

    def forward(self, x):
        assert x.shape[-1] == self.in_features
        new_shape = (*x.shape[:-1], self.out_features)
        return self.mock_gradient_sink * torch.zeros(new_shape)


class ActivationMock:
    def __init__(self, activation):
        self.__class__ = type(
            activation.__class__.__name__, (self.__class__, activation.__class__), {}
        )
        self.__dict__ = activation.__dict__

    def forward(self, x):
        return x


class Pool2dMock:
    def __init__(self, obj):
        self.__class__ = type(
            obj.__class__.__name__, (self.__class__, obj.__class__), {}
        )
        self.__dict__ = obj.__dict__
        self.kernel_size = tupleize(obj.kernel_size)
        self.stride = tupleize(obj.stride)
        self.padding = tupleize(obj.padding)
        self.dilation = tupleize(obj.dilation)
        self.mock_gradient_sink = torch.ones(1, requires_grad=True)

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (
            in_height
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
            + self.stride[0]
        ) // self.stride[0]
        out_width = (
            in_width
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
            + self.stride[1]
        ) // self.stride[1]
        return self.mock_gradient_sink * torch.zeros(
            (batch_size, in_channels, out_height, out_width)
        )


class EmbeddingMock:
    def __init__(self, obj):
        self.__class__ = type(
            obj.__class__.__name__, (self.__class__, obj.__class__), {}
        )
        self.__dict__ = obj.__dict__
        self.embedding_dim = obj.embedding_dim
        self.mock_gradient_sink = torch.ones(1, requires_grad=True)

    def forward(self, x):
        new_shape = tuple(list(x.shape) + [self.embedding_dim])
        return self.mock_gradient_sink * torch.zeros(new_shape)


def tupleize(d):
    if isinstance(d, int):
        return (d, d)
    elif isinstance(d, tuple):
        return d
    else:
        raise ValueError


builtin_mocks = {
    torch.nn.Conv2d: Conv2dMock,
    torch.nn.BatchNorm2d: Norm2dMock,
    torch.nn.Linear: LinearMock,
    torch.nn.MaxPool2d: Pool2dMock,
    torch.nn.AvgPool2d: Pool2dMock,
    torch.nn.Embedding: EmbeddingMock,
}

no_shape_change = [
    torch.nn.ReLU,
    torch.nn.ELU,
    torch.nn.LeakyReLU,
    torch.nn.GELU,
    torch.nn.Tanh,
    torch.nn.Sigmoid,
    torch.nn.Mish,
    torch.nn.Softmax,
    torch.nn.Softmin,
    torch.nn.Softmax2d,
    torch.nn.SiLU,
    torch.nn.Dropout,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
]

for activation in no_shape_change:
    builtin_mocks[activation] = ActivationMock


def mock_recursive(torch_module, extra_mocks):
    mock_dict = {**builtin_mocks, **extra_mocks}
    unimplemented_modules = set()
    for key, submodule in torch_module._modules.items():
        if submodule.__class__ in mock_dict:
            torch_module._modules[key] = mock_dict[submodule.__class__](submodule)
        elif len(submodule._modules) > 0:
            unimplemented_modules.update(mock_recursive(submodule, extra_mocks))
        else:
            unimplemented_modules.add(submodule.__class__)
    return unimplemented_modules


def mock(torch_module, debug=False, extra_mocks={}):
    unimplemented_modules = mock_recursive(torch_module, extra_mocks)
    if debug:
        print()
        for module in unimplemented_modules:
            print(module)
