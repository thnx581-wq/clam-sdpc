"""Local feature encoders. Weights remain outside this repository."""
from pathlib import Path
import timm
import torch
from torchvision import transforms
from .timm_wrapper import TimmCNNEncoder

FEATURE_DIMS = {"resnet50_trunc": 1024, "uni_v1": 1024, "h_optimus_0": 1536}


def get_encoder(name, checkpoint=None):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if checkpoint:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if name == "resnet50_trunc":
        model = TimmCNNEncoder(checkpoint=checkpoint)
    else:
        if checkpoint is None:
            raise ValueError(f"--checkpoint is required for {name}")
        if name == "uni_v1":
            model = timm.create_model("vit_large_patch16_224", pretrained=False,
                                      init_values=1e-5, num_classes=0, dynamic_img_size=True)
        elif name == "h_optimus_0":
            model = timm.create_model(
                "vit_giant_patch14_dinov2", pretrained=False, init_values=1e-5,
                dynamic_img_size=False, num_classes=0, img_size=224,
                reg_tokens=4, no_embed_class=True,
            )
            mean = (0.707223, 0.578729, 0.703617)
            std = (0.211883, 0.230117, 0.177517)
        else:
            raise ValueError(f"Unknown model: {name}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True, assign=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return model.eval(), transform
