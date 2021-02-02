#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： XHao
# datetime： 2021/1/17 19:43 
# ide： PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： XHao
# datetime： 2021/1/11 17:50
# ide： PyCharm

import torchvision
import cv2
import torch

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE)
model.eval()

def get_prediction(img, threshold):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    img = img.to(DEVICE)
    # model的返回结果
    pred = model([img])  # pred包含了预测的边框顶点、类型和置信度
    # 预测的类型
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    # 方框的位置
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    # 置信度(注意此处分数已经按从高到低排列)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    # pred_t存放了评估得分高于阈值的最后一项的序号，因为得分已经从高到低排列，所及pred_t也即是所有评估得分高于阈值的项目总数
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    boxes, pred_cls = get_prediction(img, threshold)  # Get predictions
    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        # Write the prediction class
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return img


def start_video(path, window_name='Object Detection'):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(path)  # 打开视频流(若path=0表示开启摄像头流)
    while cap.isOpened():
        # 读取一帧数据，一帧就是一张图
        ok, frame = cap.read()
        if not ok:
            break
        frame = object_detection_api(frame, 0.8)
        # 输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)  # 延时1ms切换到下一帧图像
        if c & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # path = input("input the video path: ")
    start_video(0)
    print("End Detecting..")
