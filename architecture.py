import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    '''Splits image into tokens to be fed into network.
    
    Parameters
    ----------
    in_channels: int
        Number of image color channels.
        
    patch_size: int
        Side length of a patch in pixels. Patches are square.
        
    token_dim: int
        Token-embedding vector dimension.
    '''
    
    def __init__(self, in_channels: int, patch_size: int, token_dim: int):
        super().__init__()
        
        # For simplicity, a convolution layer is used to split images
        # into patches. However, it is technically possible to do the
        # same thing without convolutions.
        self.conv = nn.Conv2d(in_channels, token_dim, patch_size, patch_size)
        
    def forward(self, x):
        return self.conv(x).flatten(-2, -1).transpose(-2, -1)


class MLP(nn.Sequential):
    '''Multilayer perceptron (MLP) block for MLP-Mixer.
    
    Parameters
    ----------
    features: int
        Number of in/out features.
        
    hidden_dim: int
        Dimension of hidden layer. Defaults to "features" if 
        "hidden_dim" is not provided.
        
    activation: nn.Module
        Activation function used. Default is nn.GELU.
    '''
    
    def __init__(self, features: int, hidden_dim: int = None, 
                 activation: nn.Module = nn.GELU):
        super().__init__(
            nn.Linear(features, hidden_dim or features),
            activation(),
            nn.Linear(hidden_dim or features, features)
        )
        

class MixerBlock(nn.Module):
    '''Mixer block for MLP-Mixer.
    
    Parameters
    ----------
    n_tokens: int 
        Number of patches an image is split into.
    
    n_channels: int
        Number of channels (token-embedding vector dimension).
        
    token_mixing_dim: int
        Number of in/out features for token-mixing MLP.
        
    channel_mixing_dim: int
        Number of in/out features for channel-mixing MLP.
    '''
    
    def __init__(self, n_tokens: int, n_channels: int,
                 token_mixing_dim: int, channel_mixing_dim: int):
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(n_channels)
        self.layer_norm_2 = nn.LayerNorm(n_channels)
        self.token_mixer = MLP(n_tokens, token_mixing_dim)
        self.channel_mixer = MLP(n_channels, channel_mixing_dim)
        
    def forward(self, x):        
        x = self.layer_norm_1(residual := x)
        x = self.token_mixer(x.permute(0, 2, 1)).permute(0, 2, 1) + residual
        x = self.layer_norm_2(residual := x)
        return self.channel_mixer(x) + residual


class MLPMixer(nn.Module):
    '''MLP-Mixer: An all-MLP Architecture for Vision.
    
    Parameters
    ----------
    image_dim: tuple
        Image dimensions in "(channels, height, width)" tuple form.
        
    patch_size: int
        Side length of a patch in pixels. Patches are square.
        
    token_dim: int
        Token-embedding vector dimension.
        
    token_mixing_dim: int
        Number of in/out features for token-mixing MLP.
        
    channel_mixing_dim: int
        Number of in/out features for channel-mixing MLP.
    
    n_blocks: int
        Number of mixer blocks.
        
    n_classes: int
        Number of output classes.
    '''
    
    def __init__(self, image_dim: tuple, patch_size: int, token_dim: int, 
                 token_mixing_dim: int, channel_mixing_dim: int,
                 n_blocks: int, n_classes: int):
        super().__init__()
        
        channels, height, width = image_dim
        assert (height % patch_size, width % patch_size) == (0, 0), \
            'Image must be divisible by the patch size.'
        
        self.tokenizer = Tokenizer(channels, patch_size, token_dim)
        self.blocks = nn.ModuleList([
            MixerBlock((height // patch_size) * (width // patch_size), 
                       token_dim, token_mixing_dim, channel_mixing_dim)
            for _ in range(n_blocks)
        ])
        self.classifier = nn.Linear(token_dim, n_classes)
        
    def forward(self, x):
        x = self.tokenizer(x)
        
        for block in self.blocks:
            x = block(x)

        # Global average pooling followed by classifier head
        return self.classifier(x.mean(dim=1))
