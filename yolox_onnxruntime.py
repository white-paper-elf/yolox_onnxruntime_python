import cv2
import numpy as np
import onnxruntime
import argparse
import os
from PIL import ImageDraw, ImageFont

class YOLOX():
    def __init__(self, model_path, conf_thres = 0.3, device='gpu', iou_thres=0.45):
        self.conf_thres = conf_thres    # 得分阈值
        self.classes = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                        "scissors", "teddy bear", "hair drier", "toothbrush")
        self.device = device    # 设备
        self.iou_thres = iou_thres  # NMS阈值
        self.input_shape = (640, 640)   # 图像尺寸
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self._COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]   # 画框颜色
).astype(np.float32).reshape(-1, 3)

    def preprocess(self, img, input_size, swap=(2, 0, 1)):  # 图像预处理
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114   # resize
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])     # 缩放比例
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)     # 维度重新匹配
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def demo_postprocess(self, outputs, img_size, p6=False):    # 图像后处理

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        # 根据特征层的高宽生成网格点
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        # 将网格点堆叠到一起
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        # 根据网格点进行解码
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):  # 画框

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

    def multiclass_nms(self, boxes, scores, iou_thres, conf_thres, class_agnostic=True):
        """Multiclass NMS implemented in Numpy"""
        if class_agnostic:
            nms_method = self.multiclass_nms_class_agnostic
        else:
            nms_method = self.multiclass_nms_class_aware
        return nms_method(boxes, scores, iou_thres, conf_thres)

    def multiclass_nms_class_agnostic(self, boxes, scores, iou_thres, conf_thres):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > conf_thres  # 判度是否大于得分阈值
        if valid_score_mask.sum() == 0:
            return None
        # 选择满足条件的框进行下一步操作
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, iou_thres)   # 非极大值抑制处理
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms_class_aware(self, boxes, scores, iou_thres, conf_thres):
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > conf_thres
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, iou_thres)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def nms(self, boxes, scores, iou_thres):    # 非极大值抑制
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]    # 左上
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]    # 右下
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]

        return keep
    def predict(self, origin_img):     # 预测

        img, ratio = self.preprocess(origin_img, self.input_shape)  # resize图像
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = net.session.run(None, ort_inputs)   # 利用框架对图像进行处理

        predictions = self.demo_postprocess(output[0], self.input_shape)[0]  # 后处理部分
        boxes = predictions[:, :4]      # [8400, 8]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        # 得到左上右下的坐标
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio     # 框的坐标信息缩放到对应维度
        dets = self.multiclass_nms(boxes_xyxy, scores, iou_thres=self.iou_thres, conf_thres=self.conf_thres)    # 非极大值抑制处理
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            # origin_img = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,    # 画框
            #                  conf=self.conf_thres, class_names=self.classes)
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                cls_id = int(final_cls_inds[i])
                score = final_scores[i]

                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])

                color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
                text = '{}:{:.1f}%'.format(self.classes[cls_id], score * 100)
                txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                cv2.rectangle(origin_img, (x0, y0), (x1, y1), color, 2)

                txt_bk_color = (self._COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                cv2.rectangle(
                    origin_img,
                    (x0, y0 + 1),
                    (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1
                )
                cv2.putText(origin_img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return origin_img

def make_parser():     # 参数管理器->参数设置
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument("-d", "--device", type=str, default="gpu", help="gpu or cpu")
    parser.add_argument("-m", "--model", type=str, default="model/yolox_l.onnx", help="Input your onnx model.")   # 模型权重
    parser.add_argument("-i", "--image_path", type=str, default='img/aaa.jpg', help="Path to your input image.")    # 预测图像的路径
    parser.add_argument("-o", "--output_dir", type=str, default='output', help="Path to your output directory.")   # 输出结果保存文件夹
    parser.add_argument("-s", "--conf_thres", type=float, default=0.3, help="Score threshould to filter the result.")   # 得分阈值
    parser.add_argument("--iou_thres", type=float, default=0.45, help="nms threshould")                                 # NMS阈值
    parser.add_argument("--with_p6", action="store_true", help="Whether your model uses p6 in FPN/PAN.")                # 是否使用p6
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    net = YOLOX(model_path=args.model, conf_thres=args.conf_thres, device=args.device, iou_thres=args.iou_thres)

    origin_img = cv2.imread(args.image_path)    # 读取图像

    origin_img = net.predict(origin_img)   # 预测图像及画框

    if not os.path.exists(args.output_dir):     # 创建结果保存文件夹
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))  # 输出路径
    cv2.imwrite(output_path, origin_img)    # 储存输出结果
