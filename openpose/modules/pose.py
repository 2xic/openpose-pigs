import cv2
import numpy as np
from openpose.modules.one_euro_filter import OneEuroFilter
import math

BODY_PARTS_KPT_IDS = [
        [
          0,
          3
        ],
        [
          1,
          3
        ],
        [
          2,
          3
        ],
        [
          3,
          4
        ],
        [
          4,
          5
        ]
      ]
BODY_PARTS_PAF_IDS = BODY_PARTS_KPT_IDS

class Pose:
    kpt_names = [
        "nose",
        "ear_left",
        "ear_right",
        "neck",
        "back",
        "tail"
    ]
    num_kpts = len(kpt_names) - 1
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img, scale_x=1, scale_y=1):
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0,255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
        for part_id in range(len(BODY_PARTS_PAF_IDS)):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]


            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                x_a *= scale_x
                y_a *= scale_y

            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                x_b *= scale_x
                y_b *= scale_y


            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), colors[part_id], 2)


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt

