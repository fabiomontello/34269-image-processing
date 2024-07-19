import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from utils import FeatureExtractor
from vit_pytorch.vit import Attention, FeedForward


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, patch_size, in_chans, dropout=0.0):
        super().__init__()

        self.decoder_embed = nn.Linear(768, dim, bias=True)
        self.norm_layer = nn.LayerNorm
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 196, dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=heads,
                    qkv_bias=True,
                    norm_layer=self.norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.decoder_norm = self.norm_layer(dim)
        self.decoder_pred = nn.Linear(dim, patch_size**2 * in_chans, bias=True)

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


class ColorNet(nn.Module):
    def __init__(self, backbone_path=None, out_channels=2, optimize_backbone=False):
        super(ColorNet, self).__init__()
        self.emb_dim = 768
        self.depth = 1
        self.heads = 8
        self.patch_size = 16
        self.out_channels = out_channels
        self.base_weights = True if backbone_path is None else False
        self.optimize_backbone = optimize_backbone

        backbone = timm.create_model(
            "vit_base_patch16_224.mae",
            pretrained=self.base_weights,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.backbone = FeatureExtractor(backbone)

        if not self.base_weights:
            self.backbone.load_state_dict(torch.load(backbone_path))

        self.decoder_r = Decoder(
            self.emb_dim, self.depth, self.heads, self.patch_size, 1
        )
        self.decoder_g = Decoder(
            self.emb_dim, self.depth, self.heads, self.patch_size, 1
        )
        self.decoder_b = Decoder(
            self.emb_dim, self.depth, self.heads, self.patch_size, 1
        )

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def forward(self, x):
        y = self.backbone(x)
        r = self.decoder_r(y)
        g = self.decoder_g(y)
        b = self.decoder_b(y)
        r = self.unpatchify(r)
        g = self.unpatchify(g)
        b = self.unpatchify(b)

        y = torch.cat((r, g, b), dim=1)
        return y
