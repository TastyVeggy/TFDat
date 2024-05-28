import multiprocessing
import tflite_runtime.interpreter as tflite
import numpy as np
from tfdat.detector.letterbox import LetterBox
import cv2
from loguru import logger


class Detector:
    def __init__(self, weights=None, use_onnx=False, mlmodel=False, use_cuda=True):
        interpreter = tflite.Interpreter(
            weights, num_threads=multiprocessing.cpu_count()
        )
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

    def detect(
        self,
        image: list,
        input_shape: tuple = (640, 640),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        filter_classes: bool = None,
        agnostic_nms: bool = True,
        with_p6: bool = False,
        return_image=False,
    ) -> list:
        original_image = image.copy()
        im = [LetterBox()(image=image)]

        im = np.stack(im)
        im = im[..., ::-1].transpose(
            (0, 1, 2, 3)
        )  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)

        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype(np.float32)
        im /= 255

        # Prepare the input tensor
        input_data = im
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Run inference
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        output_data[0][:4] *= input_shape[0]

        bs = output_data.shape[0]  # batch size
        nc = output_data.shape[1] - 4  # number of classes
        nm = output_data.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = np.amax(output_data[:, 4:mi], 1) > conf_thres  # candidates

        multi_label = False
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = np.transpose(output_data, (0, -1, -2))

        prediction[..., :4] = self._xywh2xyxy(prediction[..., :4])
        output = np.zeros((0, 6 + nm), dtype=np.float32)

        max_nms = 30000
        agnostic = False
        max_wh = 7680

        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box = x[:, :4]
            cls = x[:, 4 : 4 + nc]
            mask = x[:, 4 + nc : 4 + nc + nm]

            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)

            # Concatenate the arrays along axis 1
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)

            # Reshape conf to a 1-dimensional array
            conf_flat = conf.flatten()

            # Filter the resulting array based on the condition conf_flat > conf_thres
            filtered_x = x[conf_flat > conf_thres]

            n = filtered_x.shape[0]  # number of boxes

            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                # Sort x based on the 5th column in descending order
                sorted_indices = np.argsort(x[:, 4])[::-1]
                # Select the top max_nms rows based on the sorted indices
                x = x[sorted_indices[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            # Apply NMS using cv2.dnn.NMSBoxes function
            i = cv2.dnn.NMSBoxes(
                boxes, scores, score_threshold=0.4, nms_threshold=iou_thres
            )
            i = i[:max_det]  # limit detections

            output = x[i]

        output[:, :4] = self._scale_boxes(
            input_shape, output[:, :4], original_image.shape[:2]
        )
        logger.info(output)
        image_info = {
            "width": original_image.shape[1],
            "height": original_image.shape[0],
        }
        return output, image_info

    def _xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def _clip_boxes(self, boxes, shape):
        """
        It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
        shape

        Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image
        """
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def _scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                            calculated based on the size difference between the two images.

        Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )  # gain  = old / new
            pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
                (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self._clip_boxes(boxes, img0_shape)
        return boxes
