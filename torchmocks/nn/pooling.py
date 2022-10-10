import torch


def tupleize(d):
    if isinstance(d, int):
        return (d, d)
    elif isinstance(d, tuple):
        return d
    else:
        raise ValueError


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


mock_dict = {
    # torch.nn.modules.pooling.AvgPool1d,
    torch.nn.modules.pooling.AvgPool2d: Pool2dMock,
    # torch.nn.modules.pooling.AvgPool3d,
    # torch.nn.modules.pooling.MaxPool1d,
    torch.nn.modules.pooling.MaxPool2d: Pool2dMock,
    # torch.nn.modules.pooling.MaxPool3d,
    # torch.nn.modules.pooling.MaxUnpool1d,
    # torch.nn.modules.pooling.MaxUnpool2d,
    # torch.nn.modules.pooling.MaxUnpool3d,
    # torch.nn.modules.pooling.FractionalMaxPool2d,
    # torch.nn.modules.pooling.FractionalMaxPool3d,
    # torch.nn.modules.pooling.LPPool1d,
    # torch.nn.modules.pooling.LPPool2d,
    # torch.nn.modules.pooling.AdaptiveMaxPool1d,
    # torch.nn.modules.pooling.AdaptiveMaxPool2d,
    # torch.nn.modules.pooling.AdaptiveMaxPool3d,
    # torch.nn.modules.pooling.AdaptiveAvgPool1d,
    # torch.nn.modules.pooling.AdaptiveAvgPool2d,
    # torch.nn.modules.pooling.AdaptiveAvgPool3d,
}
