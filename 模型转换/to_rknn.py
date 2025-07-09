import os
import numpy as np
import cv2
from rknn.api import RKNN

# Constants
ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'
QUANTIZE_ON = True
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 320
CLASSES = ('helmet','nohelmet','reflective','noreflective')

# Functions
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = input.shape[0:2]

    box_confidence = input[..., 4:5]
    box_class_probs = input[..., 5:]
    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE / grid_h)

    box_wh = np.square(input[..., 2:4] * 2)
    box_wh *= anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)
    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for i, (output, mask) in enumerate(zip(input_data, masks)):
        output = output.transpose(1, 2, 0).reshape((IMG_SIZE // (8 * 2**i), IMG_SIZE // (8 * 2**i), 3, -1))
        b, c, s = process(output, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    if not boxes:
        return None, None, None

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for cls in set(classes):
        idxs = np.where(classes == cls)
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]
        keep = nms_boxes(cls_boxes, cls_scores)
        nboxes.append(cls_boxes[keep])
        nclasses.append(np.array([cls] * len(keep)))
        nscores.append(cls_scores[keep])

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def draw(img, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        left, top, right, bottom = map(int, box)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f'{CLASSES[cl]} {score:.2f}'
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# Main execution
if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    print('--> Load ONNX model')
    if rknn.load_onnx(model=ONNX_MODEL) != 0:
        exit('Load ONNX model failed!')

    print('--> Build model')
    if rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET) != 0:
        exit('Build model failed!')

    print('--> Export RKNN model')
    if rknn.export_rknn(RKNN_MODEL) != 0:
        exit('Export RKNN model failed!')

    print('--> Init runtime environment')
    if rknn.init_runtime() != 0:
        exit('Init runtime failed!')

    # Load and preprocess image
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, _, _ = letterbox(img, new_shape=IMG_SIZE)
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    print('--> Run inference')
    outputs = rknn.inference(inputs=[input_data])

    boxes, classes, scores = yolov5_post_process(outputs)

    if boxes is not None:
        img_draw = img_resized.copy()
        draw(img_draw, boxes, scores, classes)
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        cv2.imshow("Result", img_draw)
        cv2.waitKey(0)

    rknn.release()

