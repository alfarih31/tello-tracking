import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(38, 8, SSDBoxSizes(25, 57), [2]),
    SSDSpec(19, 16, SSDBoxSizes(57, 111), [2, 3]),
    SSDSpec(19, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(19, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(10, 128, SSDBoxSizes(213, 264), [2]),
    SSDSpec(10, 300, SSDBoxSizes(264, 315), [2])
]

priors = generate_ssd_priors(specs, image_size)
