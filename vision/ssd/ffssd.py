import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

class FFSSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, fusion_layer_indexes: List[int],
                 classification_headers: nn.ModuleList, fusion: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(FFSSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.fusion_layer_indexes = fusion_layer_indexes
        self.fusion = fusion
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        fusion_index = 0

        for i, index in enumerate(self.fusion_layer_indexes):
            if isinstance(index, int):
                end_layer_index = index
                inplace = False
            elif isinstance(index, list):
                end_layer_index = index[0]
                inplace=True

            for layer in self.base_net[start_layer_index:end_layer_index]:
                x = layer(x)
            y = x
            if inplace:
                y = self.fusion[fusion_index](y)
                fusion_index += 1
                y += self.fusion[fusion_index](x)
                fusion_index += 1
                y = self.fusion[fusion_index](y)
                fusion_index += 1

                x = self.base_net[end_layer_index](x)
            else:
                y = self.fusion[fusion_index](y)
                fusion_index += 1

                x = self.base_net[index](x)

                y += self.fusion[fusion_index](x)
                fusion_index += 1
                y = self.fusion[fusion_index](y)
                fusion_index += 1
            confidence, location = self.compute_header(header_index, y)
            confidences.append(confidence)
            locations.append(location)
            header_index += 1
            start_layer_index = end_layer_index+1

        for layer in self.base_net[-3:-1]:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            confidences.append(confidence)
            locations.append(location)
            header_index += 1
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        location = self.regression_headers[i](x)
        confidence = self.classification_headers[i](location)

        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers")
                                                                or k.startswith("regression_headers")
                                                                or k.startswith("extras"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)


    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
