import numpy as np

from mmdet.datasets.coco import CocoDataset
COCO_CLASSES = CocoDataset.METAINFO["classes"]


def postprocess_mmdetection_output(output, threshold = 0.2):
    scores = output.pred_instances.scores.numpy()
    boxes = output.pred_instances.bboxes.numpy()
    labels = output.pred_instances.labels.numpy() 
    
    idx_good_boxes = np.where(scores > threshold)  # boxes above threshold
    good_boxes = boxes[idx_good_boxes, :][0]
    

    good_scores = scores[idx_good_boxes]
    good_labels = labels[idx_good_boxes]

    detections = []
    for box, label, score in zip(good_boxes, good_labels, good_scores):
        detections.append({"label": COCO_CLASSES[label], "box": box.tolist(), "score": score})

    return detections
