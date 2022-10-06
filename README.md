# torchmocks
Test pytorch code with minimal computational overhead.

## Problem
The computational overhead of neural networks discourages thorough testing during development and within CI/CD pipelines.

## Solution
Torchmocks replaces common building blocks (such as torch.nn.Conv2d) with replicas that only keep track of tensor shapes and device location.  This is often the only information that we need to check to ensure proper function of pytorch code.

## Install
```
pip install torchmocks
```

## Example
```python
import torch
import torchmocks
from torchvision.models import resnet152

def test_mock_resnet():
    net = resnet152()
    torchmocks.mock(net)
    image_batch = torch.zeros(4, 3, 255, 255)
    output = net(image_batch)
    assert output.shape == (4, 1000)

```

## Status
This is a work in progress and only a handful of torch modules have been mocked. Modules that have not been mocked will run their normal computation during the forward pass.
I'm also exploring other ways to do shape inference in order to mock operations that don't appear in the module tree. Let me know if you have any ideas.