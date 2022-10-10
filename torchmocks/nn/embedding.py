import torch


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


mock_dict = {
    torch.nn.modules.sparse.Embedding: EmbeddingMock,
    torch.nn.modules.sparse.EmbeddingBag: None,
}
