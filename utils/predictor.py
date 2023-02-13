import os
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt

from heydealer_vision.utils.ultralytics.augment import AugConfig
from heydealer_vision.utils.ultralytics.augment import FlipType
from heydealer_vision.utils.ultralytics.augment import TTAModule


class TRTPredictor:
    def __init__(self,
                 engine_path: Union[os.PathLike, str],
                 classes: List[str]):
        self.stride = 32  # FIXME; hard coded-stride
        self.tta = TTAModule([
            AugConfig(),
            AugConfig(scale=0.83, flip=FlipType.LEFT_RIGHT),
            AugConfig(scale=0.67),
        ], self.stride)

        self.mean = None
        self.std = None
        self.n_classes = len(classes)
        self.class_names = classes

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def predict(self, img: np.ndarray) -> List[np.ndarray]:
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def infer_sample(self,
                     img: np.ndarray):
        img, ratio = preproc(img, self.imgsz, self.mean, self.std)
        data = self.predict(img)

        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]],
                               np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

        return dets

    def infer_sample_augmented(self,
                               img: np.ndarray) -> np.ndarray:
        dets_list = []
        for scale in [1, 0.83, 0.67]:
            input_, ratio = preproc_augment(img, self.imgsz, self.mean, self.std, scale=scale)
            data = self.predict(input_)

            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]],
                                   np.array(final_scores)[:num[0]].reshape(-1, 1),
                                   np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            dets_list.append(dets)
        merged_dets = np.concatenate(dets_list, axis=0)  # (A*N, 6)

        # nms
        if merged_dets.shape[0] > 1:
            boxes = merged_dets[:, :4]
            # FIXME: only works for single class
            scores = merged_dets[:, 4:5]
            merged_dets = multiclass_nms(boxes,
                                         scores,
                                         nms_thr=0.5,
                                         score_thr=0.25)

        return merged_dets

    def inference(self,
                  img_path: str,
                  augment: bool = False,
                  visualize: bool = False,
                  conf: float = 0.25) -> np.ndarray:
        """
        return: (# dets, 6) -> [...box, score, cls_index]
            if there is no detection, the return shape is (0, 6)
        """
        origin_img = cv2.imread(img_path)

        if augment:
            dets = self.infer_sample_augmented(origin_img)
        else:
            dets = self.infer_sample(origin_img)

        if not visualize:
            return dets

        if dets is None or dets.shape[0] == 0:
            return origin_img

        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=conf, class_names=self.class_names)
        return origin_img

    def get_fps(self):
        import time
        img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.predict(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.predict(img)
        print(100 / (time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
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

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image: np.ndarray,
            input_size: List[int],
            mean: Optional[float],
            std: Optional[float],
            swap: Tuple[int, ...] = (2, 0, 1)) -> Tuple[np.ndarray, float]:
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    if input_size[0] == img.shape[0] and input_size[1] == img.shape[1]:
        resized_img = img
    else:
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def preproc_augment(image: np.ndarray,
                    input_size: List[int],
                    mean: Optional[float],
                    std: Optional[float],
                    scale: float = 1.,
                    swap: Tuple[int, ...] = (2, 0, 1)) -> Tuple[np.ndarray, float]:
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)

    scaled_input_size = [int(s * scale) for s in input_size]
    r = min(scaled_input_size[0] / img.shape[0], scaled_input_size[1] / img.shape[1])
    if scaled_input_size[0] == img.shape[0] and scaled_input_size[1] == img.shape[1]:
        resized_img = img
    else:
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n / size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
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

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            # (x0, y0 + 1),
            # (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            (x0, y0 - int(1.5 * txt_size[1])),
            (x0 + txt_size[0] + 1, y0 + 1),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - int(0.5 * txt_size[1])), font, 0.4, txt_color, thickness=1)
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
