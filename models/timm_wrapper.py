"""CLAM's truncated ResNet50 (third residual stage, pooled to 1024 values)."""
import timm
import torch


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = timm.create_model(
            "resnet50.tv_in1k", features_only=True, out_indices=(3,),
            pretrained=checkpoint is None, num_classes=0,
        )
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or any(not key.startswith(("layer4.", "fc.")) for key in unexpected):
                raise ValueError(f"ResNet checkpoint mismatch: {missing}, {unexpected}")
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, batch):
        return self.pool(self.model(batch)[0]).flatten(1)
