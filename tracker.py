"""
high level support for doing this and that.
"""

import cv2
from vision.utils import Timer
from utils import Flags

timer = Timer()

class Matcher:
    """
        Matcher will get matched frame with initial frame and marked by a bbox.
    This object must be initialized by the initial frame, i.e. the template.
        The marked bbox will be used for tracking the same object then by comparing
    the intersect area of all box in detected bboxes.
        The __call__  will return the marked bbox which has been shrinked by 60%. It is needed
    to get shrinked because sometimes there's another nearby detector's bboxes which intersect each other.
    The shrinked marked bbox will be the most intersected with the supposed bbox (the target).
            Return: (xmin, ymin, w, h)
    """
    def __init__(self, template):
        self.template = template
        self.pick = lambda image: cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)

    def __call__(self, img):
        h, w = self.template.shape[:]
        h, w = h*0.6, w*0.6
        res = self.pick(img)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        return (max_loc[0]+w/6, max_loc[1]+h/6, max_loc[0]+w, max_loc[1]+h)

class Tracker:
    def __init__(self, kw: float, kh: float, predictor):
        self.predictor = predictor
        self.flags = Flags()
        self.kw = kw
        self.kh = kh
        self.matcher = None
        self.target_box = None
        self.dists = None

    def __call__(self, img):
        """
        Return: (Distance by h, Distance by w, image)
        """
        boxes = self.detect(img)

        if not self.flags.tracker_init:
            self.init_tracker(img, boxes)

        return self.track(img, boxes)

    def init_tracker(self, img, boxes):
        """
        Used to initializing the Matcher object with the most nearest object
        """
        # Get box with the most largetst area, i.e. the most nearest object
        minArea = 0
        for i in range(boxes.size(0)):
            h, w = boxes[i, :][:-1]
            area = h * w
            if area > minArea:
                box = boxes[i, :]
                minArea = area
        self.matcher = Matcher(img[int(box[1]):int(box[1]+box[3]),
                                   int(box[0]):int(box[0]+box[2])])
        self.flags.tracker_init = True

    def track(self, img, boxes):
        """
        Used to keep an eye to target by picking the bbox with most largest
        intersect area with marked bbox (target's bbox).
            Return: (Distance by h, Distance by w, image)
        """
        # Get target's bbox by matcher object
        track_loc = self.matcher(img)
        # Filter box in boxes which has highest intersect area
        minIntersect = 0
        for i in range(boxes.size(0)):
            intersect = self.compute_intersect(track_loc, boxes[i, :])
            if intersect > minIntersect:
                self.target_box = boxes[i, :]
                minIntersect = intersect
        self.dists = self.get_distance(*self.target_box[:-1])
        cv2.rectangle(img,
                      (int(self.target_box[0]), int(self.target_box[1])),
                      (int(self.target_box[0] + self.target_box[2]), int(self.target_box[1] + self.target_box[3])),
                      (255, 0, 0), 2)
        for idx, dist in enumerate(self.dists):
            cv2.putText(img, dist, (self.target_box[0]+20, self.target_box[1]+idx*20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), 2)
        return img

    def detect(self, image):
        """
        Inference the detector to detect all human object in images
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, _ = self.predictor.predict(img, 10, 0.4)
        interval = timer.end()
        if boxes.size(0) == 0:
            raise ValueError("%s: There's no detected object"%self.detect.__name__)

        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        return boxes

    def compute_intersect(self, boxA, boxB, eps=1e-5):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        intersect = max(0, xB - xA + 1 + eps) * max(0, yB - yA + 1 + eps)
        return intersect

    def get_distance(self, h, w):
        return (self.kh/h, self.kw/w)
