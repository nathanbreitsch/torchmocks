import torch


class MockLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        out_features, in_features = weight.shape
        assert x.shape[-1] == in_features
        output_shape = (*x.shape[:-1], out_features)
        output = torch.zeros(output_shape, requires_grad=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros(x.shape, requires_grad=True)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros(weight.shape, requires_grad=True)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros(bias.shape, requires_grad=True)
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


mock_dict = {
    torch.nn.modules.linear.Identity: None,
    torch.nn.modules.linear.Linear: LinearModuleMock,
    # torch.nn.modules.linear.Bilinear,
    # torch.nn.modules.linear.LazyLinear,
}
