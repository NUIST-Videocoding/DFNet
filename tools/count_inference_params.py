#!/usr/bin/env python3
"""Count parameters in the enhancement/fusion modules used by RCNN inference.

Run from the repository root with the project's Python environment:
    python tools/count_inference_params.py
"""

from detectron2.modeling.meta_arch.rcnn import Lenhance, MSFR, all_Fusion


def count_parameters(module):
    """Return (all parameters, trainable parameters) for a module."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main():
    # Keep these constructor arguments in sync with GeneralizedRCNN.__init__.
    modules = {
        "lenhance": Lenhance(256, 256),
        "all_fusion_l": all_Fusion(256, 256, 192),
        "msfr": MSFR(192),
    }

    totals = 0
    trainable_totals = 0
    print(f"{'component':<16} {'total params':>16} {'trainable params':>19}")
    print("-" * 53)
    for name, module in modules.items():
        total, trainable = count_parameters(module)
        totals += total
        trainable_totals += trainable
        print(f"{name:<16} {total:>16,} {trainable:>19,}")

    print("-" * 53)
    print(f"{'total':<16} {totals:>16,} {trainable_totals:>19,}")
    print(f"\nTotal: {totals / 1_000_000:.6f} M parameters")
    print(f"Trainable: {trainable_totals / 1_000_000:.6f} M parameters")


if __name__ == "__main__":
    main()
