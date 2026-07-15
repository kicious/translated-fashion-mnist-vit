import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=1,  num_classes=10,
                  embed_dim=128,depth=4,num_heads=4,mlp_ratio=4.0):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        #Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)

        #Class Token & Absolute Positional Encoding
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,self.num_patches+1,embed_dim) * 0.02)

        #Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        #Classification Head
        self.head = nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        B = x.shape[0]

        #Patch embedding:[B,C,H,W] -> [B,embed_dim,N] -> [B,N,embed_dim]
        x = self.patch_embed(x).flatten(2).transpose(1,2)

        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens,x),dim=1)

        #Add absolute positional encoding
        x = x + self.pos_embed

        x = self.transformer(x)

        cls_feat = self.norm(x[:,0])
        out = self.head(cls_feat)

        return out
