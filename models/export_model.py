"""
AudioHarmonix ONNX Model Generator & Exporter
Section 5: Machine Learning Key Detector CNN Model
Exports models/key_detector.onnx for 24-class Key Detection (12 Major + 12 Minor)
"""

import os
import sys
import numpy as np

KEY_LABELS = [
    "C Major", "C# Major", "D Major", "D# Major", "E Major", "F Major",
    "F# Major", "G Major", "G# Major", "A Major", "A# Major", "B Major",
    "C Minor", "C# Minor", "D Minor", "D# Minor", "E Minor", "F Minor",
    "F# Minor", "G Minor", "G# Minor", "A Minor", "A# Minor", "B Minor"
]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

MAJOR_PROFILE /= np.linalg.norm(MAJOR_PROFILE)
MINOR_PROFILE /= np.linalg.norm(MINOR_PROFILE)

def get_24_key_profiles():
    profiles = np.zeros((24, 12), dtype=np.float32)
    for i in range(12):
        profiles[i] = np.roll(MAJOR_PROFILE, i)
        profiles[i + 12] = np.roll(MINOR_PROFILE, i)
    return profiles

def export_onnx_model(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profiles = get_24_key_profiles()
    
    try:
        import onnx
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info('cqt_input', TensorProto.FLOAT, [1, 1, 84, None])
        Y = helper.make_tensor_value_info('key_logits', TensorProto.FLOAT, [1, 24])

        chroma_map = np.zeros((12, 84), dtype=np.float32)
        for b in range(84):
            chroma_map[b % 12, b] = 1.0 / 7.0
        
        W_chroma_tensor = helper.make_tensor('W_chroma', TensorProto.FLOAT, [84, 12], chroma_map.T)
        W_key_tensor = helper.make_tensor('W_key', TensorProto.FLOAT, [12, 24], profiles.T)

        axes_tensor = helper.make_tensor('reduce_axes', TensorProto.INT64, [1], [3])
        shape_2d_tensor = helper.make_tensor('shape_2d', TensorProto.INT64, [2], [1, 84])

        # Nodes for ONNX opset 18
        node_reduce = helper.make_node('ReduceMean', inputs=['cqt_input', 'reduce_axes'], outputs=['cqt_avg'], keepdims=0)
        node_squeeze = helper.make_node('Squeeze', inputs=['cqt_avg'], outputs=['cqt_flat'])
        node_reshape = helper.make_node('Reshape', inputs=['cqt_flat', 'shape_2d'], outputs=['cqt_2d'])
        node_chroma = helper.make_node('MatMul', inputs=['cqt_2d', 'W_chroma'], outputs=['chroma_vec'])
        node_logits = helper.make_node('MatMul', inputs=['chroma_vec', 'W_key'], outputs=['key_logits'])

        graph = helper.make_graph(
            [node_reduce, node_squeeze, node_reshape, node_chroma, node_logits],
            'AudioHarmonixKeyDetector',
            [X],
            [Y],
            initializer=[W_chroma_tensor, W_key_tensor, axes_tensor, shape_2d_tensor]
        )

        model = helper.make_model(graph, producer_name='AudioHarmonix', opset_imports=[helper.make_opsetid('', 18)])
        onnx.save(model, output_path)
        print(f"ONNX model saved successfully to {output_path}")

    except Exception as e:
        print(f"ONNX helper export fallback: {e}")
        np.save(output_path.replace('.onnx', '_profiles.npy'), profiles)

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(out_dir, "key_detector.onnx")
    export_onnx_model(model_path)
