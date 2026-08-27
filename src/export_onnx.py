"""
ONNX Export and INT8 Quantization Module.
Converts trained PyTorch model to ONNX and performs CPU INT8 quantization for real-time inference.
"""

import os
import torch
import numpy as np
from src.config import ModelConfig, ONNX_DIR, CHECKPOINT_DIR
from src.models.efficientnet_b0 import EfficientNetB0Baseline

def export_model_to_onnx(
    checkpoint_path: str = None,
    output_onnx_path: str = None,
    quantize: bool = True
) -> str:
    """
    Exports EfficientNetB0Baseline PyTorch model to ONNX format and applies INT8 quantization.
    """
    if output_onnx_path is None:
        output_onnx_path = os.path.join(ONNX_DIR, "efficientnet_b0_baseline.onnx")
        
    model_cfg = ModelConfig()
    model = EfficientNetB0Baseline(model_cfg, pretrained=False)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Exporting model with initial weights (no checkpoint specified).")
        
    model.eval()
    
    # Dummy input: (1, 1, 224, 224)
    dummy_input = torch.randn(1, 1, 224, 224, dtype=torch.float32)
    
    # Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_lfcc'],
        output_names=['output_logits'],
        dynamic_axes={
            'input_lfcc': {0: 'batch_size'},
            'output_logits': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported PyTorch model to ONNX: {output_onnx_path}")
    
    quant_onnx_path = output_onnx_path.replace(".onnx", "_int8.onnx")
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=output_onnx_path,
                model_output=quant_onnx_path,
                weight_type=QuantType.QUInt8
            )
            print(f"Successfully performed INT8 dynamic quantization: {quant_onnx_path}")
            return quant_onnx_path
        except Exception as e:
            print(f"INT8 quantization notice ({e}). Returning FP32 ONNX model.")
            return output_onnx_path
            
    return output_onnx_path

if __name__ == '__main__':
    export_model_to_onnx()
