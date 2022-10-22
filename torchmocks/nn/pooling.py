import torch


class MockPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, dilation, padding, stride):
        ctx.save_for_backward(torch.IntTensor(tuple(x.shape)))

        batch_size = x.shape[0]
        in_channels = x.shape[1]
        spacial_shape = x.shape[2:]
        spacial_dim = len(spacial_shape)
        out_channels = in_channels
        new_spacial_shape = [
            (
                spacial_shape[i]
                + 2 * padding[i]
                - dilation[i] * (kernel_size[i] - 1)
                - 1
                + stride[i]
            )
            // stride[i]
            for i in range(spacial_dim)
        ]
        new_shape = tuple([batch_size, out_channels] + new_spacial_shape)
        return torch.zeros(new_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_shape = tuple(ctx.saved_tensors[0])
        return torch.zeros(x_shape), None, None, None, None


def tupleize(d, dim=2):
    if isinstance(d, int):
        return tuple([d] * dim)
    elif isinstance(d, tuple):
        return d
    else:
        raise ValueError


class MockPoolModule:
    def __init__(self, obj):
        super().__init__()
        self.__class__ = type(
            obj.__class__.__name__, (self.__class__, obj.__class__), {}
        )
        self.__dict__ = obj.__dict__
        self.kernel_size = tupleize(obj.kernel_size)
        self.dilation = tupleize(getattr(obj, "dilation", 1))
        self.padding = tupleize(obj.padding)
        self.stride = tupleize(obj.stride)

    def forward(self, x):
        return MockPoolFunction.apply(
            x,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )


mock_dict = {
    # torch.nn.modules.pooling.AvgPool1d,
    torch.nn.modules.pooling.AvgPool2d: MockPoolModule,
    # torch.nn.modules.pooling.AvgPool3d,
    # torch.nn.modules.pooling.MaxPool1d,
    torch.nn.modules.pooling.MaxPool2d: MockPoolModule,
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
