import sys
import predict

try:
    path = "test_crop.png"
    model_path = predict.find_best_model()
    from ultralytics import YOLO
    model = YOLO(model_path)
    boxes, img = predict.detect_gates(path, model)
    num_labels, labels_im = predict.trace_wires(img, boxes)
    graph, global_inputs, global_outputs = predict.build_graph(boxes, num_labels, labels_im, img.shape)
    
    with open('debug_out.txt', 'w') as f:
        f.write("Inputs: " + str(global_inputs) + "\n\n")
        f.write("Graph:\n")
        for g in graph:
            f.write(f"{g}: {graph[g]}\n")
            
except Exception as e:
    import traceback
    with open('debug_out.txt', 'w') as f:
        f.write(traceback.format_exc())
