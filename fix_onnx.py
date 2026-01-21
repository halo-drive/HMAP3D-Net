import onnx
from onnx import helper, numpy_helper
import numpy as np

def fix_1d_outputs(input_path, output_path):
    """Fix 1D output tensors to be 2D for DriveWorks compatibility"""
    model = onnx.load(input_path)
    graph = model.graph
    
    print("Fixing 1D outputs...")
    
    # Find the 'iou' output and its producer node
    iou_output = None
    iou_idx = None
    iou_producer_node = None
    
    for i, output in enumerate(graph.output):
        if output.name == 'iou':
            iou_output = output
            iou_idx = i
            break
    
    if iou_output is None:
        print("ERROR: 'iou' output not found!")
        return
    
    # Get current shape
    shape = [d.dim_value for d in iou_output.type.tensor_type.shape.dim]
    print(f"  Found 'iou': {shape}")
    
    if len(shape) == 1:
        print(f"  Reshaping to {shape + [1]}")
        
        # Find the node that produces 'iou'
        for node in graph.node:
            for output_name in node.output:
                if output_name == 'iou':
                    iou_producer_node = node
                    break
            if iou_producer_node:
                break
        
        if iou_producer_node:
            # Rename the original output to an intermediate name
            new_intermediate_name = 'iou_1d_intermediate'
            
            # Update the producer node's output
            for i, output_name in enumerate(iou_producer_node.output):
                if output_name == 'iou':
                    iou_producer_node.output[i] = new_intermediate_name
                    break
            
            # Create new shape constant [100, 1]
            new_shape_name = "iou_reshape_shape"
            new_shape_value = np.array([shape[0], 1], dtype=np.int64)
            
            shape_tensor = helper.make_tensor(
                name=new_shape_name,
                data_type=onnx.TensorProto.INT64,
                dims=[2],
                vals=new_shape_value.tolist()
            )
            graph.initializer.append(shape_tensor)
            
            # Add Reshape node: iou_1d_intermediate → iou (reshaped to 2D)
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[new_intermediate_name, new_shape_name],
                outputs=['iou'],
                name='iou_reshape_to_2d'
            )
            graph.node.append(reshape_node)
            
            # Update the graph output tensor shape to [100, 1]
            # Clear existing dimensions
            while len(iou_output.type.tensor_type.shape.dim) > 0:
                iou_output.type.tensor_type.shape.dim.pop()
            
            # Add new dimensions [100, 1]
            dim1 = iou_output.type.tensor_type.shape.dim.add()
            dim1.dim_value = shape[0]
            
            dim2 = iou_output.type.tensor_type.shape.dim.add()
            dim2.dim_value = 1
            
            print("  ✓ Fixed!")
        else:
            print("  ERROR: Could not find producer node for 'iou'")
            return
    
    # Save
    onnx.save(model, output_path)
    print(f"\nSaved fixed model to: {output_path}")
    
    # Verify
    print("\nVerification:")
    fixed_model = onnx.load(output_path)
    for output in fixed_model.graph.output:
        if output.name == 'iou':
            shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
            status = "✓" if len(shape) >= 2 else "✗"
            print(f"  'iou' is now: {shape} ({len(shape)}D) {status}")
    
    # Check that model is valid
    try:
        onnx.checker.check_model(fixed_model)
        print("\n✓ Model is valid!")
    except Exception as e:
        print(f"\n✗ Model validation failed: {e}")

# Run the fix
fix_1d_outputs("./stage2_3d_headsstatic.onnx", "./stage2_3d_headsstatic_fixed.onnx")
