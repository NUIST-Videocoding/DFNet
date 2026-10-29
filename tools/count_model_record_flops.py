#!/usr/bin/env python3
"""Count FLOPs for a model archived as a standalone meta_arch/rcnn.py file.

Default target: ../model_record/ours_y2.s2/rcnn.py
Run from the repository root in the project's CUDA environment:
    python tools/count_model_record_flops.py
"""

import argparse
import sys
import types
from pathlib import Path

import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis

# Reuse the FLOPs conventions and custom operator handles from the main script.
from count_inference_flops import add_custom_flop_handles


DEFAULT_MODEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "model_record"
    / "ours_y2.s2"
    / "rcnn.py"
)


def load_rcnn_module(source_path):
    """Load archived rcnn.py classes without registering duplicate meta-archs."""
    source = source_path.read_text(encoding="utf-8")
    source = source.replace("@META_ARCH_REGISTRY.register()", "")

    module_name = "detectron2.modeling.meta_arch.rcnn_model_record_flops"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = "detectron2.modeling.meta_arch"
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


class ArchivedInferenceFeaturePath(nn.Module):
    def __init__(self, archived_rcnn):
        super().__init__()
        self.lenhance = archived_rcnn.Lenhance(256, 256)
        self.all_fusion_l = archived_rcnn.all_Fusion(256, 256, 192)
        self.iwt = archived_rcnn.IWT()
        self.msfr = archived_rcnn.MSFR(192)

    def forward(self, features):
        p2, p3, p4, p5 = self.lenhance(features)
        fuse_l = self.all_fusion_l(p2, p3, p4, p5)

        # Match the archived rcnn.py inference path, where high-frequency
        # features are currently replaced with zeros before IWT.
        fuse_h = torch.zeros_like(fuse_l).repeat(1, 3, 1, 1)
        fused = self.iwt(torch.cat((fuse_l, fuse_h), dim=1))
        height, width = features["p5"].shape[-2:]
        fused = fused[:, :, :height, :width]
        pyramid = self.msfr(fused)
        return tuple(pyramid[name] for name in ("p2", "p3", "p4", "p5", "p6"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="Archived rcnn.py to analyze",
    )
    args = parser.parse_args()
    source_path = args.model_file.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Model source not found: {source_path}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This path requires CUDA: archived IWT allocates its output with "
            ".cuda(), and the Mamba selective scan requires CUDA."
        )

    archived_rcnn = load_rcnn_module(source_path)
    device = torch.device("cuda")
    model = ArchivedInferenceFeaturePath(archived_rcnn).to(device).eval()
    sizes = {"p2": (200, 200), "p3": (100, 100), "p4": (50, 50), "p5": (25, 25)}
    features = {
        name: torch.randn(1, 256, height, width, device=device)
        for name, (height, width) in sizes.items()
    }

    with torch.no_grad():
        analysis = add_custom_flop_handles(FlopCountAnalysis(model, (features,)))
        total_flops = analysis.total()

    print(f"Model source: {source_path}")
    print("FLOPs scope: archived Lenhance + all_fusion_L + IWT + MSFR")
    print("Excluded: backbone, net_belle (not called by inference), RPN, ROI heads")
    print("Input feature sizes (batch=1, channels=256):")
    for name, (height, width) in sizes.items():
        print(f"  {name}: {height}x{width}")
    print(f"Total: {total_flops / 1e9:.6f} G FLOPs ({total_flops:,.0f} FLOPs)")

    unsupported = analysis.unsupported_ops()
    if unsupported:
        print("Operators still not covered by a FLOPs rule:")
        for operator, count in sorted(unsupported.items()):
            print(f"  {operator}: {count} call(s)")
    else:
        print("All traced operators are covered by the configured FLOPs rules.")
    print("Note: SelectiveScanFn uses the approximate 9*B*D*N*L FLOPs convention.")


if __name__ == "__main__":
    main()
