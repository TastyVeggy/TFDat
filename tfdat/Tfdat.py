from tfdat.tracker import Tracker
from tfdat.detector import Detector
from tfdat.utils.default_cfg import config
import os
from loguru import logger
import time
import tfdat.utils as utils
import cv2
import copy


class Tfdat:
    def __init__(self, weights: str = None) -> None:

        self.detector = Detector(weights)
        self.tracker = Tracker(self.detector)

    def track_video(self, video_path, **kwargs):
        output_filename = os.path.basename(video_path)
        kwargs["filename"] = output_filename
        config = self._update_args(kwargs)

        for bbox_details, frame_details in self._start_tracking(video_path, config):
            yield bbox_details, frame_details

    def _update_args(self, kwargs):
        for key, value in kwargs.items():
            if key in config.keys():
                config[key] = value
            else:
                print(f'"{key}" argument not found! valid args: {list(config.keys())}')
                raise Exception("Above error")
        return config

    def _start_tracking(self, stream_path: str, config: dict):
        fps = config.pop("fps")
        output_dir = config.pop("output_dir")
        filename = config.pop("filename")
        save_result = config.pop("save_result")
        display = config.pop("display")
        class_names = config.pop("class_names")

        cap = cv2.VideoCapture(stream_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            logger.info(f"video save path is {save_path}")

            video_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        tic = time.time()

        prevTime = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            im0 = copy.deepcopy(frame)

            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(
                frame, config
            )

            """logger.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                                 elapsed_time * 1000))"""

            """if self.recognizer:
                res = self.recognizer.recognize(im0, horizontal_list=bboxes_xyxy,
                            free_list=[])
                im0 = utils.draw_text(im0, res)
            else:
                im0 = utils.draw_boxes(im0,
                                    bboxes_xyxy,
                                    class_ids,
                                    identities=ids,
                                    draw_trails=draw_trails,
                                    class_names=class_names)"""

            im0 = draw_boxes(
                im0, bboxes_xyxy, class_ids, identities=ids, class_names=class_names
            )

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            """cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            if display:
                cv2.imshow('Testing', im0)"""
            if save_result:
                video_writer.write(im0)

            frame_id += 1

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

            # yeild required values in form of (bbox_details, frames_details)
            yield (bboxes_xyxy, ids, scores, class_ids), (
                im0 if display else frame,
                frame_id - 1,
                fps,
            )

        tac = time.time()
        print(f"Total Time Taken: {tac - tic:.2f}")


def draw_boxes(img, bbox_xyxy, class_ids, identities=None, class_names=None):

    for i, box in enumerate(bbox_xyxy):
        # get ID of object
        id = int(identities[i]) if identities is not None else None

        # if class_ids is not None:
        color = utils.compute_color_for_labels(int(class_ids[i]))

        obj_name = class_names[int(class_ids[i])]

        label = f"{id}: {obj_name}"

        utils.draw_ui_box(box, img, label=label, color=color, line_thickness=2)

    return img
