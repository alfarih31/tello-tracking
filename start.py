from djitellopy import Tello
import cv2
from time import sleep
from json import load
from numpy import array
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
from simple_pid import PID

import argparse

label_path = 'models/landing.txt'
model_path = 'models/landing10000.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--tracking', type=str, default='ssd', help='Tracking algorithm to use: ssd or algo', required=True)
opt = parser.parse_args()

tello = Tello()
tello.connect()
battery = tello.get_battery()
tello.streamon()
sleep(2)
frame_read = tello.get_frame_read()
h, w = frame_read.frame.shape[:2]
set_point_x = w/2 #Image x's Center
set_point_y = h/2-60 #Image y's Center
lambda_x = 0.125*set_point_x #Target x tolerance
lambda_y = 0.125*set_point_y
set_point_dist = 120
lambda_d = 0.15*set_point_dist

kw = 2375
kh = 38950

with open('calibrateData.json') as jsonFile:
    data = load(jsonFile)
    cameraMat = data['cameraMat']
    distortCoef = data['distortionCoef']
cameraMat = array(cameraMat)
distortCoef = array(distortCoef)

class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)


timer = Timer()

def control(key, speed):
    if key == ord('z'): return (tello.takeoff(), speed)
    elif key == ord('x'): return (tello.land(), speed)
    elif key == ord('a'): return (tello.move_left(50), speed)
    elif key == ord('d'): return (tello.move_right(50), speed)
    elif key == ord('w'): return (tello.move_forward(50), speed)
    elif key == ord('s'): return (tello.move_back(50), speed)
    elif key == ord('j'): return (tello.rotate_counter_clockwise(20), speed)
    elif key == ord('l'): return (tello.rotate_clockwise(20), speed)
    elif key == ord('i'): return (tello.move_up(20), speed)
    elif key == ord('k'): return (tello.move_down(20), speed)
    elif key == ord('[') and speed > 10:
        speed -= 10
        return (tello.set_speed(speed), speed)
    elif key == ord(']') and speed < 100:
        speed += 10
        return (tello.set_speed(speed), speed)
    return (True, speed)

def detect(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, _ = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}'.format(interval*1000, labels.size(0)))
    if boxes.size(0) == 1:
        box = boxes[0, :]
        _box = (box[0], box[1], box[2]-box[0], box[3]-box[1])
        return (True, _box)
    else: return (False, False)

def get_distance(wp, hp):
    return (kw/wp, kh/hp)

class Tracker():
    def __init__(self, algo='ssd', frame = False, bbox = False):
        self.tracker = None
        if algo == 'algo':
            assert type(frame) != bool, 'Frame Cannot be empty for Algo'
            assert type(bbox) != bool, 'Bbox cannot be None for Algo'
            self._tracker = cv2.TrackerMedianFlow_create()
            self._tracker.init(frame, bbox)
            self.tracker = self._tracker.update
        elif algo == 'ssd':
            self.tracker = detect

        assert self.tracker != None, 'Wrong Algo'
    def __call__(self, frame):
        return self.tracker(frame)

def maintain_x(out):
    if out >= 0: tello.rotate_counter_clockwise(int(3.58*out+2.0))
    elif out < 0: tello.rotate_clockwise(int(-3.58*out+2.0))

def maintain_y(out):
    if out >= 0: tello.move_up(int(4.8*out+20.0))
    elif out < 0: tello.move_down(int(-4.8*out+20.0))

def maintain_dist(out):
    if out >= 0: tello.move_back(int(4.8*out+20.0))
    elif out < 0: tello.move_forward(int(-4.8*out+20.0))

def init_control(x_flag, y_flag, outx, outy):
    if not x_flag: maintain_x(outx)
    elif not y_flag: maintain_y(outy)

def program():
    pid_x = PID(0.022, 0.0005, 0.001, setpoint=set_point_x, output_limits=(-100, 100), sample_time=None)
    pid_y = PID(0.01, 0.0000, 0.0005, setpoint=set_point_y, output_limits=(-100, 100), sample_time=None)
    pid_dist = PID(0.007, 0.0001, 0.004, setpoint=set_point_dist, output_limits=(-100, 100), sample_time=None)
    fail = 0
    mv_y = 0
    mv_x = 0
    mv_dist = 0
    x_flag = False
    y_flag = False
    dist_flag = False
    init_flag = False
    buff2 = 'No Accomplished'
    wait = 1 if opt.tracking == 'ssd' else 200
    while True:
        _frame = frame_read.frame
        frame = cv2.undistort(_frame, cameraMat, distortCoef)
        ok, bbox = detect(frame)
        cv2.putText(frame, 'PROGRAM',
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 0),
                    2)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            r1 = bbox[3]/bbox[2]
            r2 = bbox[2]/bbox[3]
            dr1 = abs(r1-1.673)
            dr2 = abs(r2-0.597)
            if dr1 <= 0.3 or dr2 <= 0.2: break
        else: fail += 1

        if fail > 5:
            tello.rotate_clockwise(10)
            fail = 0
        cv2.imshow('TELLO', frame)
        cv2.waitKey(30)
    if opt.tracking == 'algo': tracker = Tracker(algo=opt.tracking, frame=_frame, bbox=bbox)
    else: tracker = Tracker(algo=opt.tracking)
    while True:
        frame = cv2.undistort(frame_read.frame, cameraMat, distortCoef)
        ok, bbox = tracker(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

            center_x = (bbox[2]/2)+bbox[0]
            center_y = (bbox[3]/2)+bbox[1]
            err_x = abs(center_x-set_point_x)
            err_y = abs(center_y-set_point_y)

            d1, d2 = get_distance(bbox[2], bbox[3])
            err_dist = abs(d2-set_point_dist)
            if not init_flag:
                if not x_flag:
                    if (err_x < lambda_x):
                        x_flag = True
                        pid_x.reset()
                        buff2 = 'X Centered'
                    elif (err_x > lambda_x): mv_x = pid_x(center_x)
                    buff = 'Controlling center X %.2f, Current Center X %.2f'%(mv_x, center_x)
                elif not y_flag:
                    if (err_y < lambda_y):
                        y_flag = True
                        pid_y.reset()
                        buff2 = buff2 + '  Y Centered'
                    elif (err_y > lambda_y): mv_y = pid_y(center_y)
                    buff = 'Controlling center Y %.2f, Current Center Y%.2f'%(mv_y, center_y)
                init_control(x_flag, y_flag, mv_x, mv_y)

                if x_flag and y_flag: init_flag = True
            else:
                if not dist_flag:
                    if (err_dist < lambda_d):
                        dist_flag = True
                        pid_dist.reset()
                        buff2 = buff2 + '  Dist Approached'
                    elif (err_dist > lambda_d):
                        mv_dist = pid_dist(d2)
                        maintain_dist(mv_dist)
                        if err_x > lambda_x: x_flag = False, pid_dist.reset()
                        elif err_y > lambda_y: y_flag = False, pid_dist.reset()
                    buff = 'Controlling Distance %.2f, Current Dist %.2f'%(mv_dist, d2)
            cv2.putText(frame, buff,
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (200, 0, 255),
                1)

        cv2.putText(frame, buff2,
                (20, int(h/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 255, 0),
                2)
        cv2.imshow('TELLO', frame)
        key = cv2.waitKey(wait)
        if key == 27: break

def main():
    count = 0
    _speed = 10
    while True:
        image = frame_read.frame
        cv2.imshow('TELLO', image)
        key = cv2.waitKey(30)
        count += 1
        if count == 17:
            flag, speed = control(key, _speed)
            if flag: _speed = speed
            b = tello.get_battery()
            print(b)
            count = 0

        if key == 27: break
        elif key == ord('p'): program()

if __name__ == '__main__':
    main()
    tello.land()
    sleep(1)
    tello.end()
