import torch


class MockConv2d:
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
        return torch.zeros((batch_size, self.out_channels, out_height, out_width))


class MockBatchNorm:
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


class MockLinear:
    def __init__(self, linear):
        self.__class__ = type(
            linear.__class__.__name__, (self.__class__, linear.__class__), {}
        )
        self.__dict__ = linear.__dict__
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        assert x.shape[-1] == self.in_features
        new_shape = list(x.shape)
        new_shape[-1] = self.out_features
        return torch.zeros(new_shape)


class MockActivation:
    def __init__(self, activation):
        self.__class__ = type(
            activation.__class__.__name__, (self.__class__, activation.__class__), {}
        )
        self.__dict__ = activation.__dict__

    def forward(self, x):
        return x


class MaxPool2dMock:
    def __init__(self, obj):
        self.__class__ = type(
            obj.__class__.__name__, (self.__class__, obj.__class__), {}
        )
        self.__dict__ = obj.__dict__
        self.kernel_size = tupleize(obj.kernel_size)
        self.stride = tupleize(obj.stride)
        self.padding = tupleize(obj.padding)
        self.dilation = tupleize(obj.dilation)

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
        return torch.zeros((batch_size, in_channels, out_height, out_width))


def tupleize(d):
    if isinstance(d, int):
        return (d, d)
    elif isinstance(d, tuple):
        return d
    else:
        raise ValueError


mock_dict = {
    torch.nn.Conv2d: MockConv2d,
    torch.nn.BatchNorm2d: MockBatchNorm,
    torch.nn.ReLU: MockActivation,
    torch.nn.Linear: MockLinear,
    torch.nn.MaxPool2d: MaxPool2dMock,
}


def mock_recursive(torch_module):
    unimplemented_modules = set()
    for key, submodule in torch_module._modules.items():
        if submodule.__class__ in mock_dict:
            torch_module._modules[key] = mock_dict[submodule.__class__](submodule)
        elif len(submodule._modules) > 0:
            unimplemented_modules.update(mock_recursive(submodule))
        else:
            unimplemented_modules.add(submodule.__class__)
    return unimplemented_modules


def mock(torch_module, debug=False):
    unimplemented_modules = mock_recursive(torch_module)
    if debug:
        print()
        for module in unimplemented_modules:
            print(module)
