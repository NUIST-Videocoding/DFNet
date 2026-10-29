#!/usr/bin/env python3
"""Estimate FLOPs for the custom feature path used by RCNN inference.

Counts Lenhance -> all_fusion_L -> IWT -> MSFR for one image, with p2-p5
feature sizes 200x200, 100x100, 50x50, and 25x25 respectively.

Run from the repository root in the project's CUDA environment:
    python tools/count_inference_flops.py
"""

import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

from detectron2.modeling.meta_arch.rcnn import IWT, Lenhance, MSFR, all_Fusion


def _numel(value):
    shape = get_shape(value)
    if shape is None or any(size is None for size in shape):
        return 0
    result = 1
    for size in shape:
        result *= size
    return result


def _constant(value):
    """Read scalar/list constants from TorchScript values when possible."""
    try:
        constant = value.toIValue()
        if constant is not None:
            return constant
    except (AttributeError, RuntimeError):
        pass

    try:
        node = value.node()
        if node.kind() in ("prim::ListConstruct", "prim::TupleConstruct"):
            return [_constant(item) for item in node.inputs()]
    except (AttributeError, RuntimeError):
        pass
    return None


def _elementwise(inputs, outputs):
    # Count one FLOP per output element for add/sub/mul/div and activations.
    return _numel(outputs[0])


def _reduction(inputs, outputs):
    # Approximate sum/mean as one accumulation per input element; include the
    # final division for mean.
    return _numel(inputs[0]) + (_numel(outputs[0]) if inputs and outputs else 0)


def _adaptive_max_pool(inputs, outputs):
    in_shape = get_shape(inputs[0])
    out_shape = get_shape(outputs[0])
    if not in_shape or not out_shape or len(in_shape) < 2:
        return 0
    in_spatial = in_shape[-2] * in_shape[-1]
    out_spatial = out_shape[-2] * out_shape[-1]
    outer = _numel(inputs[0]) // max(in_spatial, 1)
    return outer * max(in_spatial - out_spatial, 0)


def _max_pool(inputs, outputs):
    shape = get_shape(outputs[0])
    kernel = _constant(inputs[1]) if len(inputs) > 1 else None
    if shape is None or kernel is None:
        return 0
    if isinstance(kernel, int):
        kernel_area = kernel * kernel
    else:
        kernel_area = 1
        for size in kernel:
            kernel_area *= size
    return _numel(outputs[0]) * max(kernel_area - 1, 0)


def _selective_scan(inputs, outputs):
    """Approximate Mamba selective-scan FLOPs using its standard 9*B*D*N*L estimate."""
    u_shape = get_shape(inputs[0])
    a_shape = get_shape(inputs[2])
    if not u_shape or not a_shape or len(u_shape) != 3 or len(a_shape) != 2:
        return 0
    batch, channels, sequence = u_shape
    state_size = a_shape[-1]
    return 9 * batch * channels * state_size * sequence


def add_custom_flop_handles(analysis):
    # fvcore's standard counter covers convolutions/linears, but these ops are
    # used heavily by attention, wavelet reconstruction, and Mamba in this path.
    handles = {
        "aten::add": _elementwise,
        "aten::add_": _elementwise,
        "aten::sub": _elementwise,
        "aten::mul": _elementwise,
        "aten::mul_": _elementwise,
        "aten::div": _elementwise,
        "aten::neg": _elementwise,
        "aten::abs": _elementwise,
        "aten::exp": _elementwise,
        "aten::sigmoid": _elementwise,
        "aten::silu": _elementwise,
        "aten::gelu": _elementwise,
        "aten::leaky_relu": _elementwise,
        "aten::leaky_relu_": _elementwise,
        "aten::sqrt": _elementwise,
        "aten::pow": _elementwise,
        "aten::mean": _reduction,
        "aten::sum": _reduction,
        "aten::adaptive_max_pool2d": _adaptive_max_pool,
        "aten::max_pool2d": _max_pool,
        # Data movement/rearrangement operators do not perform arithmetic.
        "aten::clone": lambda inputs, outputs: 0,
        "aten::pad": lambda inputs, outputs: 0,
        "aten::repeat": lambda inputs, outputs: 0,
        "aten::pixel_shuffle": lambda inputs, outputs: 0,
        "prim::PythonOp.LowerBoundFunction": _elementwise,
        "prim::PythonOp.SelectiveScanFn": _selective_scan,
    }
    analysis.set_op_handle(**handles)
    # Mamba's in_proj/dt_proj parameters are read directly in einsums, rather
    # than through their Linear.forward methods; their parent FLOPs are counted.
    analysis.uncalled_modules_warnings(False)
    return analysis


class InferenceFeaturePath(nn.Module):
    """The enhancement path actually called by GeneralizedRCNN.inference."""

    def __init__(self):
        super().__init__()
        self.lenhance = Lenhance(256, 256)
        self.all_fusion_l = all_Fusion(256, 256, 192)
        self.iwt = IWT()
        self.msfr = MSFR(192)

    def forward(self, features):
        p2, p3, p4, p5 = self.lenhance(features)
        fuse_l = self.all_fusion_l(p2, p3, p4, p5)

        # Match rcnn.py inference: high-frequency bands are currently zeros.
        fuse_h = torch.zeros_like(fuse_l).repeat(1, 3, 1, 1)
        fused = self.iwt(torch.cat((fuse_l, fuse_h), dim=1))
        h, w = features["p5"].shape[-2:]
        fused = fused[:, :, :h, :w]
        pyramid = self.msfr(fused)

        # Return tensors so the tracer sees the complete path's outputs.
        return tuple(pyramid[name] for name in ("p2", "p3", "p4", "p5", "p6"))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This inference path requires CUDA: IWT in rcnn.py allocates its "
            "output with .cuda(). Run this script in the project's CUDA environment."
        )

    device = torch.device("cuda")
    model = InferenceFeaturePath().to(device).eval()
    sizes = {"p2": (200, 200), "p3": (100, 100), "p4": (50, 50), "p5": (25, 25)}
    features = {
        name: torch.randn(1, 256, height, width, device=device)
        for name, (height, width) in sizes.items()
    }

    with torch.no_grad():
        analysis = add_custom_flop_handles(FlopCountAnalysis(model, (features,)))
        total_flops = analysis.total()

    print("FLOPs scope: Lenhance + all_fusion_L + IWT + MSFR")
    print("Excluded: backbone, net_belle (not called by inference), RPN, ROI heads")
    print("Input feature sizes (batch=1, channels=256):")
    for name, (height, width) in sizes.items():
        print(f"  {name}: {height}x{width}")
    print(f"Total: {total_flops / 1e9:.6f} G FLOPs ({total_flops:,.0f} FLOPs)")

    unsupported = analysis.unsupported_ops()
    if unsupported:
        print("\nOperators still not covered by a FLOPs rule:")
        for operator, count in sorted(unsupported.items()):
            print(f"  {operator}: {count} call(s)")
        print("Review these operators before treating the total as complete.")
    else:
        print("All traced operators are covered by the configured FLOPs rules.")
    print("Note: SelectiveScanFn uses the approximate 9*B*D*N*L FLOPs convention.")


if __name__ == "__main__":
    main()
