# tello-tracking
DJI Ryze Tello tracking project for my last assignment

# Pre
## Requirements
    1. Python3 installed
    2. PIP
    3. A trained mobilenet-v2 ssd model from
        https://github.com/qfgaohao/pytorch-ssd

## Initiating

1. Preparing required package

        pip install -r requirements.txt

2. Locate model path and labels path for SSD detection

        Edit model_path and label_path variable at start.py line 12, 13

# Start Tracking!!!

        - For tracking using SSD detection:
            python3 start.py --tracking ssd
        - For tracking using MedianFlow tracker:
            pythn3 start.py --tracking algo

    ## How to:

        - After Tello window's show up
        - Press 'z' to take off
        - Press 'p' to start tracking and approaching to target which have trained in ssd model
