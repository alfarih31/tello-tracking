from djitellopy import Tello
from json import load
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
from numpy import array
import cv2
from time import sleep, time
import csv

tello = Tello()
tello.connect()
battery = tello.get_battery()
tello.streamon()
sleep(2)
frame_read = tello.get_frame_read()
h, w = frame_read.frame.shape[:2]
_cx = w/2 #Image x's Center
_cy = h/2-120 #Image y's Center

timer = Timer()
with open('calibrateData.json') as jsonFile:
    data = load(jsonFile)
    cameraMat = data['cameraMat']
    distortCoef = data['distortionCoef']
cameraMat = array(cameraMat)
distortCoef = array(distortCoef)

label_path = 'models/voc-model-labels.txt'
model_path = 'models/landing10000.pth'
class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)


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
    _image = cv2.undistort(image, cameraMat, distortCoef)
    image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}'.format(interval*1000, labels.size(0)))
    if boxes.size(0) == 1:
        box = boxes[0, :]
        label = f"{class_names[labels[0]]}: {probs[0]:.2f}"
        cv2.rectangle(_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
        return (_image, box)
    else: return (_image, False)

def tuning():
    cxs = []
    times = []
    while True:
        image = frame_read.frame
        image, box = detect(image)
        if type(box) != bool:
            break
    while True:
        tello.rotate_clockwise(3)
        image, bbox = detect(frame_read.frame)
        cv2.imshow('TUNING', image)
        if type(bbox) != bool:
            cx = (bbox[2]+bbox[0])/2
            cxs.append(cx)
            times.append(time())
        key = cv2.waitKey(1)
        if key == 27: break
    with open('dataTuning.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        for cx, t in zip(cxs, times):
            csvwriter.writerow([t, cx.item()])

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
        elif key == ord('t'): tuning()


if __name__ == '__main__':
    tello.set_speed(10)
    main()
    tello.land()
    sleep(1)
    tello.end()
