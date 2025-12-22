from types import MethodType

import torch
import torch.nn.functional as F
from cellpose.vit_sam import Transformer
from segment_anything.modeling.image_encoder import get_rel_pos
from torch import nn

from classpose.log import get_logger
from classpose.unet import UNet

models_logger = get_logger(__name__)


def flash_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Function replacing the forward pass of the attention layer to use the
    flash attention implementation.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    B, H, W, _ = x.shape
    L = H * W

    # qkv shape: (3, B, num_heads, L, head_dim)
    qkv = (
        self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)

    attn_mask = None
    if getattr(self, "use_rel_pos", False):
        head_dim = q.shape[-1]
        q_hw = q.reshape(B, self.num_heads, H, W, head_dim)

        Rh = get_rel_pos(H, H, self.rel_pos_h)  # (H, H, head_dim)
        Rw = get_rel_pos(W, W, self.rel_pos_w)  # (W, W, head_dim)

        # rel_h, rel_w: (B, num_heads, H, W, H)
        rel_h = torch.einsum("b n h w c, h k c -> b n h w k", q_hw, Rh)
        rel_w = torch.einsum("b n h w c, w k c -> b n h w k", q_hw, Rw)

        # reshape to: (B, num_heads, L, L)
        bias = (rel_h[..., :, None] + rel_w[..., None, :]).reshape(
            B, self.num_heads, L, L
        )
        attn_mask = bias

    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=self.scale,
    )  # (B, num_heads, L, head_dim)

    x = x.transpose(1, 2).reshape(B, H, W, -1)
    x = self.proj(x)
    return x


def patch_attention_forwards(
    model: nn.Module, attn_class_name: str = "Attention"
):
    n = 0
    for module in model.modules():
        if module.__class__.__name__ == attn_class_name:
            module.forward = MethodType(flash_forward, module)
            n += 1
    models_logger.info(
        f"Patched {n} {attn_class_name} modules with scaled_dot_product_attention"
    )


class ClassTransformer(Transformer):
    def __init__(
        self,
        backbone: str = "vit_l",
        ps: int = 8,
        nout: int = 3,
        bsize: int = 256,
        rdrop: float = 0.4,
        checkpoint: str | None = None,
        n_cell_classes: int = 1,
        feature_transformation_structure: list[str] | None = None,
    ):
        """
        Initialize the ClassTransformer.

        Args:
            backbone (str, optional): Name of the backbone to use. Defaults to
                "vit_l".
            ps (int, optional): Patch size. Defaults to 8.
            nout (int, optional): Number of output channels. Defaults to 3.
            bsize (int, optional): Batch size. Defaults to 256.
            rdrop (float, optional): random layer dropout (for training).
                Defaults to 0.4.
            checkpoint (str | None, optional): Path to the checkpoint. Defaults
                to None.
            n_cell_classes (int, optional): Number of cell classes. Defaults to
                1.
            feature_transformation_structure (list[str] | None, optional):
                Feature transformation structure. Defaults to None.
        """
        super(ClassTransformer, self).__init__(
            backbone=backbone,
            ps=ps,
            nout=nout,
            bsize=bsize,
            rdrop=rdrop,
            checkpoint=checkpoint,
        )
        self.n_cell_classes = n_cell_classes
        self.feature_transformation_structure = feature_transformation_structure

        # The classpose extension only makes sense if we are detecting more than one class
        if n_cell_classes > 1:
            if feature_transformation_structure is not None:
                self.out_class = UNet(
                    in_channels=256,
                    out_channels=self.n_cell_classes * ps**2,
                    n_channels=feature_transformation_structure,
                )
            else:
                # we maintain here the core nomenclature used by Cellpose to avoid complications :-)
                self.out_class = nn.Conv2d(
                    256, self.n_cell_classes * ps**2, kernel_size=1
                )
            self.W3 = nn.Parameter(
                torch.eye(self.n_cell_classes * ps**2).reshape(
                    self.n_cell_classes * ps**2, self.n_cell_classes, ps, ps
                ),
                requires_grad=False,
            )

        patch_attention_forwards(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor.
                - Random features (kept for consistency with Cellpose 3).
        """
        x = self.encoder.patch_embed(x)

        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        if self.training and self.rdrop > 0:
            nlay = len(self.encoder.blocks)
            rdrop = (
                torch.rand((len(x), nlay), device=x.device)
                < torch.linspace(0, self.rdrop, nlay, device=x.device)
            ).float()
            for i, blk in enumerate(self.encoder.blocks):
                mask = rdrop[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x * mask + blk(x) * (1 - mask)
        else:
            for blk in self.encoder.blocks:
                x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        # readout is changed here
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride=self.ps, padding=0)

        # Classpose extension: add the second output for cell classification
        if self.n_cell_classes > 1:
            x2 = self.out_class(x)
            x2 = (
                F.conv_transpose2d(x2, self.W3, stride=self.ps, padding=0)
                if self.n_cell_classes > 1
                else None
            )
            # not super intuitive but it is what it is
            out = torch.cat((x2, x1), 1)
        else:
            out = x1

        return out, torch.randn((x.shape[0], 256), device=x.device)

    def freeze_backbone(self):
        """
        Freeze the backbone of the model.
        """
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_instance_classification(self):
        """
        Freeze the instance classification head of the model.
        """
        self.out.eval()
        for param in self.out.parameters():
            param.requires_grad = False
        self.W2.requires_grad = False

    def freeze_neck(self):
        """
        Freeze the neck of the model.
        """
        self.encoder.neck.eval()
        for param in self.encoder.neck.parameters():
            param.requires_grad = False

    def unfreeze_neck(self):
        """
        Unfreeze the neck of the model.
        """
        self.encoder.neck.train()
        for param in self.encoder.neck.parameters():
            param.requires_grad = True

    def freeze(
        self,
        backbone: bool = True,
        instance_classification: bool = True,
        neck: bool = True,
    ):
        """
        Freeze the backbone and/or the instance classification head of the model.
        Helpful for fine-tuning.
        """
        if backbone:
            self.freeze_backbone()
        if instance_classification:
            self.freeze_instance_classification()
        if neck:
            self.freeze_neck()
        else:
            self.unfreeze_neck()

    def load_classification_head(self, pretrained_path: str):
        """
        Load a pretrained classification head from a checkpoint. Allows for
        missing keys in the checkpoint.

        Args:
            pretrained_path (str): Path to the checkpoint.
        """
        checkpoint = torch.load(pretrained_path)
        if "out" in checkpoint:
            self.out.load_state_dict(checkpoint["out"])
        if "out_class" in checkpoint:
            self.out_class.load_state_dict(checkpoint["out_class"])
        if "W2" in checkpoint:
            self.W2.data = checkpoint["W2"]
        if "W3" in checkpoint:
            self.W3.data = checkpoint["W3"]

    def save_model(
        self, filename: str, save_only_trainable_params: bool = False
    ):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
            save_only_trainable_params (bool, optional): Whether to save only
                trainable parameters. Defaults to False.
        """
        state_dict = self.state_dict()
        if save_only_trainable_params:
            for n, p in self.named_parameters():
                if p.requires_grad is False:
                    state_dict.pop(n)
        torch.save(state_dict, filename)
