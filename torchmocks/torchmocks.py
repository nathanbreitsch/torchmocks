import torch

from .nn import activation, normalization, linear, conv, pooling, embedding, dropout


builtin_mocks = {
    **embedding.mock_dict,
    **pooling.mock_dict,
    **linear.mock_dict,
    **conv.mock_dict,
    **dropout.mock_dict,
    **normalization.mock_dict,
    **activation.mock_dict,
}


def mock_recursive(torch_module, extra_mocks):
    mock_dict = {**builtin_mocks, **extra_mocks}
    unimplemented_modules = set()
    for key, submodule in torch_module._modules.items():
        if mock_dict.get(submodule.__class__) is not None:
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
