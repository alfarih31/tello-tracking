import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import MobileNetV2

from .ffssd import FFSSD, GraphPath
from .predictor import Predictor
from .config import mobilenetv2_ffssd_config as config

def activate_sum(onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return ReLU()


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, use_batch_norm=True,onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                groups=in_channels, stride=stride, padding=padding),
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )
    else:
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                groups=in_channels, stride=stride, padding=padding),
            ReLU(),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

def Separabledeconv(inp, oup, kernel, padding=0, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        # dw
        nn.ConvTranspose2d(inp, inp, kernel, 2, padding, groups=inp, bias=False),
        BatchNorm2d(inp),
        ReLU(),

        # pw
        Conv2d(inp, oup, 1, bias=False),
        BatchNorm2d(oup),
    )

def create_mobilenetv2_ffssd(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm,
                           onnx_compatible=onnx_compatible).features
    fusion_layer_indexes = [7, [10], [12], 14]

    fusion = ModuleList([
        #1 19 to 38
        Sequential(
            #pw
            Conv2d(32, 64, 1, 1, bias=False),
            BatchNorm2d(64),
        ),
        Separabledeconv(64, 64, 2),
        Sequential(
            activate_sum(),
        ),
        #2 19 Inplace
        Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1, bias=False),
            BatchNorm2d(64),
        ),
        Sequential(
            # Only DW
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, #To half size
                    stride=2, padding=1, groups=64, bias=False),
            BatchNorm2d(64),
            nn.ReLU6(),
            # dw
            nn.ConvTranspose2d(64, 64, 3, 2, 1, groups=64, bias=False),
            BatchNorm2d(64),
        ),
        Sequential(
            activate_sum(),
        ),
        #3 19 Inplace
        Sequential(
            Conv2d(in_channels=96, out_channels=96, kernel_size=3,
                stride=1, padding=1, bias=False),
            BatchNorm2d(96),
        ),
        Sequential(
            # Only DW
            Conv2d(in_channels=96, out_channels=96, kernel_size=3, #To half size
                    stride=2, padding=1, groups=96, bias=False),
            BatchNorm2d(96),
            nn.ReLU6(),
            # dw
            nn.ConvTranspose2d(96, 96, 3, 2, 1, groups=96, bias=False),
            BatchNorm2d(96),
        ),
        Sequential(
            activate_sum(),
        ),
        #4 10 to 19
        Sequential(
            #pw
            Conv2d(96, 160, 1, 1, bias=False),
            BatchNorm2d(160),
        ),
        Separabledeconv(160, 160, 3, 1),
        Sequential(
            activate_sum(),
        ),
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=64, out_channels=4 * 4, kernel_size=3, padding=1,), #38 5776
        SeperableConv2d(in_channels=64, out_channels=6 * 4, kernel_size=3, padding=1,), #19 1444
        SeperableConv2d(in_channels=96, out_channels=6 * 4, kernel_size=3, padding=1,), #19 2166
        SeperableConv2d(in_channels=160, out_channels=6 * 4, kernel_size=3, padding=1,), #19 2166
        SeperableConv2d(in_channels=160, out_channels=4 * 4, kernel_size=3, padding=1,), #10 600
        SeperableConv2d(in_channels=320, out_channels=4 * 4, kernel_size=3, padding=1,), #10 600
    ])

    classification_headers = ModuleList([
        Sequential(
            Conv2d(in_channels=4 * 4, out_channels=64, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv2d(in_channels=64, out_channels=4 * num_classes, kernel_size=1),
        ),
        Sequential(
            Conv2d(in_channels=6 * 4, out_channels=64, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ),
        Sequential(
            Conv2d(in_channels=6 * 4, out_channels=96, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv2d(in_channels=96, out_channels=6 * num_classes, kernel_size=1),
        ),
        Sequential(
            Conv2d(in_channels=6 * 4, out_channels=160, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv2d(in_channels=160, out_channels=6 * num_classes, kernel_size=1),
        ),
        Sequential(
            Conv2d(in_channels=4 * 4, out_channels=160, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv2d(in_channels=160, out_channels=4 * num_classes, kernel_size=1),
        ),
        Sequential(
            Conv2d(in_channels=4 * 4, out_channels=320, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv2d(in_channels=320, out_channels=4 * num_classes, kernel_size=1),
        ),
    ])

    return FFSSD(num_classes, base_net, fusion_layer_indexes,
               classification_headers, fusion, regression_headers, is_test=is_test, config=config)


def create_mobilenetv2_ffssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
