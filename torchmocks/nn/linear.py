import torch


class MockLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(
            torch.IntTensor(tuple(x.shape)),
            torch.IntTensor(tuple(weight.shape)),
            torch.IntTensor(tuple(bias.shape)) if bias is not None else None,
        )
        out_features, in_features = weight.shape
        assert x.shape[-1] == in_features
        output_shape = (*x.shape[:-1], out_features)
        output = torch.zeros(output_shape)
        return output

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
        return grad_input, grad_weight, grad_bias


class LinearModuleMock:
    def __init__(self, obj):
        super().__init__()
        self.__class__ = type(
            obj.__class__.__name__, (self.__class__, obj.__class__), {}
        )
        self.__dict__ = obj.__dict__
        self.input_features = obj.in_features
        self.output_features = obj.out_features
        self.weight = obj.weight
        self.bias = obj.bias

    def forward(self, x):
        return MockLinearFunction.apply(x, self.weight, self.bias)


def pure_linear_mock(x, weight, bias):
    return MockLinearFunction.apply(x, weight, bias)


mock_dict = {
    torch.nn.modules.linear.Identity: None,
    torch.nn.modules.linear.Linear: LinearModuleMock,
    torch.nn.functional.linear: pure_linear_mock,
    # torch.nn.modules.linear.Bilinear,
    # torch.nn.modules.linear.LazyLinear,
}
