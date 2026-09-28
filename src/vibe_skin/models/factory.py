import timm
import torch.nn as nn


def create_model(cfg: dict, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg["name"],
        pretrained=cfg.get("pretrained", True),
        num_classes=num_classes,
        drop_rate=cfg.get("drop_rate", 0.0),
    )
    # Some timm heads start with large random weights, which gives confident wrong
    # predictions at step 0. A zero head starts from uniform probabilities instead.
    head = model.get_classifier()
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classifier head."""
    head = set(id(p) for p in model.get_classifier().parameters())
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable
