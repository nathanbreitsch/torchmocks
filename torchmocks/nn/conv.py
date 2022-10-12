import torch


class MockConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, kernel_size, dilation, padding, stride):
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        ctx.save_for_backward(x, weight, bias)
        ctx.save_for_backward(
            torch.IntTensor(tuple(x.shape)),
            torch.IntTensor(tuple(weight.shape)),
            torch.IntTensor(tuple(bias.shape)) if bias is not None else None,
        )

        out_height = (
            in_height
            + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1)
            - 1
            + stride[0]
        ) // stride[0]

        out_width = (
            in_width
            + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1)
            - 1
            + stride[1]
        ) // stride[1]

        return torch.zeros((batch_size, out_channels, out_height, out_width))

    @staticmethod
    def backward(ctx, grad_output):
        x_shape, weight_shape, bias_shape = ctx.saved_tensors
        x_shape = tuple(x_shape.tolist())
        weight_shape = tuple(weight_shape.tolist())
        bias_shape = tuple(bias_shape.tolist()) if bias_shape is not None else None

        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros(x_shape)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros(weight_shape)
        if bias_shape is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros(bias_shape)
        return grad_input, grad_weight, grad_bias, None, None, None, None


class MockConvModule:
    def __init__(self, obj):
        super().__init__()
        self.__class__ = type(
            obj.__class__.__name__, (self.__class__, obj.__class__), {}
        )
        self.__dict__ = obj.__dict__
        self.weight = obj.weight
        self.bias = obj.bias
        self.in_channels = obj.in_channels
        self.out_channels = obj.out_channels
        self.kernel_size = obj.kernel_size
        self.stride = obj.stride
        self.padding = obj.padding
        self.dilation = obj.dilation

    def forward(self, x):
        return MockConvFunction.apply(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )


mock_dict = {
    # torch.nn.modules.conv.Conv1d,
    torch.nn.modules.conv.Conv2d: MockConvModule,
    # torch.nn.modules.conv.Conv3d,
    # torch.nn.modules.conv.ConvTranspose1d,
    # torch.nn.modules.conv.ConvTranspose2d,
    # torch.nn.modules.conv.ConvTranspose3d,
    # torch.nn.modules.conv.LazyConv1d,
    # torch.nn.modules.conv.LazyConv2d,
    # torch.nn.modules.conv.LazyConv3d,
    # torch.nn.modules.conv.LazyConvTranspose1d,
    # torch.nn.modules.conv.LazyConvTranspose2d,
    # torch.nn.modules.conv.LazyConvTranspose3d,
}
