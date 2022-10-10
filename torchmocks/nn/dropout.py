import torch

mock_dict = {
    torch.nn.modules.dropout.Dropout: None,
    torch.nn.modules.dropout.Dropout1d: None,
    torch.nn.modules.dropout.Dropout2d: None,
    torch.nn.modules.dropout.Dropout3d: None,
    torch.nn.modules.dropout.AlphaDropout: None,
    torch.nn.modules.dropout.FeatureAlphaDropout: None,
}
