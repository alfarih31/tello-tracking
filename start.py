"""
tellotracker:
Allows manual operation of the drone and demo tracking mode.

Requires mplayer to record/save video.

Controls:
- tab to lift off
- WASD to move the drone
- space/shift to ascend/descent slowly
- Q/E to yaw slowly
- arrow keys to ascend, descend, or yaw quickly
- backspace to land, or P to palm-land
- enter to take a picture
- R to start recording video, R again to stop recording
  (video and photos will be saved to a timestamped file in ~/Pictures/)
- Z to toggle camera zoom state
  (zoomed-in widescreen or high FOV 4:3)
- T to toggle tracking
@author Leonie Buckley, Saksham Sinha and Jonathan Byrne
@copyright 2018 see license file for details
"""
import time
import datetime
import os
from json import load as json_load

import tellopy
import av
import cv2
from yaml import load as yaml_load
from numpy import array as np_array
from torch import load, set_flush_denormal
from simple_pid import PID

from pynput import keyboard
from utils import Flags
from vision.ssd.predictor import Predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ffssd import create_mobilenetv2_ffssd, create_mobilenetv2_ffssd_predictor
from tracker import Tracker

set_flush_denormal(True)
configs = yaml_load(open('configs.yaml', 'r'))
with open(configs['CALIB_PATH'], 'r') as json:
    json_datas = json_load(json)
    camera_mat = json_datas['cameraMat']
    distort_coef = json_datas['distortionCoef']
camera_mat = np_array(camera_mat)
distort_coef = np_array(distort_coef)

def init_predictor():
    class_names = [name.strip() for name in open(configs['MODELS']['LABEL']).readlines()]
    num_classes = len(class_names)

    if configs['MODELS']['NAME'] == 'mb2-ssd':
        net = create_mobilenetv2_ssd_lite(
            num_classes=num_classes,
            is_test=True,
        )
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif configs['MODELS']['NAME'] == 'mb2-ffssd':
        net = create_mobilenetv2_ffssd(
            num_classes=num_classes,
            is_test=True,
        )
        predictor = create_mobilenetv2_ffssd_predictor(net, candidate_size=200)
    else: raise ValueError("Available model: \n 1. mb2-ffssd \n 2. mb2-ssd")

    net.load_state_dict(load(configs['MODELS']['WEIGHT'], map_location=lambda storage, loc: storage))

    return predictor

def main():
    """ Create a tello controller and show the video feed."""
    tellotrack = TelloCV(_predictor=init_predictor(), use_w=(lambda: 0, lambda: 1)[configs['MEASURING']['use'] == 'w']())

    for packet in tellotrack.container.demux((tellotrack.vid_stream,)):
        for frame in packet.decode():
            image = tellotrack.process_frame(frame)
            cv2.imshow('tello', image)
            _ = cv2.waitKey(1) & 0xFF

class TelloCV(object):
    """
    TelloTracker builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self, _predictor: Predictor, use_w: int):
        # Define all flags
        self.flags = Flags()
        self.use_w = use_w # 1 if 'use' in configs is 'w'. else 0

        # Init drone
        self.prev_flight_data = None
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.speed = 50
        self.drone = tellopy.Tello()
        self.init_drone()
        self.init_controls()
        self.pid_configs = configs['PID']
        self.mission_controls = None
        self.variables = ['x', 'y', 'd']
        self.process_mv = dict
        self.process_pv = dict
        self.tolerances = dict

        # container for processing the packets into frames
        self.container = av.open(self.drone.get_video_stream())
        self.vid_stream = self.container.streams.video[0]
        self.out_file = None
        self.out_stream = None
        self.out_name = None
        self.start_time = time.time()

        self.tracker = Tracker(kw=configs['MEASURING']['w'], kh=configs['MEASURING']['h'],
                               predictor=_predictor, flags=self.flags)

    def init_drone(self):
        """Connect, uneable streaming and subscribe to events"""
        # self.drone.log.set_level(2)
        self.drone.connect()
        self.drone.start_video()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
                             self.handle_flight_received)


    def on_press(self, keyname):
        """handler for keyboard listener"""
        if self.flags.keydown:
            return
        try:
            self.flags.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.quit()
                exit(0)
            elif keyname == 'm':
                return self.toggle_mission(self.speed)

            # Return if on mission, mean disable all key pressed execept above keys
            if self.flags.on_mission:
                return

            if keyname in self.controls:
                key_handler = self.controls[keyname]
                if isinstance(key_handler, str):
                    getattr(self.drone, key_handler)(self.speed)
                else:
                    key_handler(self.speed)
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""
        self.flags.keydown = False
        keyname = str(keyname).strip('\'')
        print('-' + keyname)
        if keyname in self.controls:
            key_handler = self.controls[keyname]
            if isinstance(key_handler, str):
                getattr(self.drone, key_handler)(0)
            else:
                key_handler(0)

    def init_controls(self):
        """Define keys and add listener"""
        self.controls = {
            'w': 'forward',
            's': 'backward',
            'a': 'left',
            'd': 'right',
            'Key.space': 'up',
            'Key.shift': 'down',
            'Key.shift_r': 'down',
            'q': 'counter_clockwise',
            'e': 'clockwise',
            'i': lambda speed: self.drone.flip_forward(),
            'k': lambda speed: self.drone.flip_back(),
            'j': lambda speed: self.drone.flip_left(),
            'l': lambda speed: self.drone.flip_right(),
            # arrow keys for fast turns and altitude adjustments
            'Key.left': lambda speed: self.drone.counter_clockwise(speed),
            'Key.right': lambda speed: self.drone.clockwise(speed),
            'Key.up': lambda speed: self.drone.up(speed),
            'Key.down': lambda speed: self.drone.down(speed),
            'Key.tab': lambda speed: self.drone.takeoff(),
            'Key.backspace': lambda speed: self.drone.land(),
            'p': lambda key_up: self.palm_land(key_up),
            'r': lambda key_up: self.toggle_recording(key_up),
            'z': lambda key_up: self.toggle_zoom(key_up),
            'm': lambda key_up: self.toggle_mission(key_up),
            'Key.enter': lambda key_up: self.take_picture(key_up)
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()

        self.pid_x = PID(*self.pid_configs['constants']['x'], output_limits=(-100, 100), sample_time=None)
        self.pid_y = PID(*self.pid_configs['constants']['y'], output_limits=(-100, 100), sample_time=None)
        self.pid_d = PID(*self.pid_configs['constants']['d'], setpoint=self.pid_configs['distance'],
                         output_limits=(-100, 100), sample_time=None)

    def init_mission_controls(self, h, w):
        self.process_mv = {
            'x': lambda mv: self.drone.counter_clockwise(mv) if mv >= 0 else self.drone.clockwise(mv),
            'y': lambda mv: self.drone.up(mv) if mv >= 0 else self.drone.down(mv),
            'd': lambda mv: self.drone.forward(mv) if mv >= 0 else self.drone.backward(mv),
        }
        self.pid_x.setpoint = w/2
        self.pid_y.setpoint = h/2
        self.process_pv = {
            'x': self.pid_x,
            'y': self.pid_y,
            'd': self.pid_d,
        }
        self.tolerances = {
            'x': self.pid_configs['lambda'][0]*self.pid_x.setpoint,
            'y': self.pid_configs['lambda'][1]*self.pid_y.setpoint,
            'd': self.pid_configs['lambda'][2]*self.pid_d.setpoint,
        }
        self.flags.mission_controls_init = True
        return

    def process_frame(self, frame):
        """convert frame to cv2 image"""
        image = cv2.cvtColor(np_array(
            frame.to_image()), cv2.COLOR_RGB2BGR)
        image = cv2.undistort(image, camera_mat, distort_coef)

        if not self.flags.mission_controls_init:
            self.init_mission_controls(*image.shape)

        if self.flags.record:
            self.record_vid(frame)

        if self.flags.on_mission:
            try:
                image = self.tracker(image)
                if not self.flags.on_tracking:
                    self.flags.on_tracking = not self.flags.on_tracking
                    self.reset_all_pids()
                self.mission_control()
            except ValueError as e:
                print(e)
                if not self.flags.on_tracking:
                    self.flags.fails_counter += 1
                    self.move_drone()

        image = self.write_hud(image)

        return image

    def write_hud(self, frame):
        """Draw drone info, tracking and record on frame"""
        stats = self.prev_flight_data.split('|')
        if self.drone.zoom:
            stats.append("VID")
        else:
            stats.append("PIC")
        if self.flags.record:
            diff = int(time.time() - self.start_time)
            mins, secs = divmod(diff, 60)
            stats.append("REC {:02d}:{:02d}".format(mins, secs))

        if self.flags.on_mission:
            stats.append("ON MISSION")

        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=30)
        return frame

    def toggle_recording(self, key_up):
        """Handle recording keypress, creates output stream and file"""
        if key_up == 0:
            return
        self.flags.record = not self.flags.record

        if self.flags.record:
            datename = [os.getenv('HOME'), datetime.datetime.now().strftime(self.date_fmt)]
            self.out_name = '{}/Pictures/tello-{}.mp4'.format(*datename)
            print("Outputting video to:", self.out_name)
            self.out_file = av.open(self.out_name, 'w')
            self.start_time = time.time()
            self.out_stream = self.out_file.add_stream(
                'mpeg4', self.vid_stream.rate)
            self.out_stream.pix_fmt = 'yuv420p'
            self.out_stream.width = self.vid_stream.width
            self.out_stream.height = self.vid_stream.height

        if not self.flags.record:
            print("Video saved to ", self.out_name)
            self.out_file.close()
            self.out_stream = None

    def record_vid(self, frame):
        """
        convert frames to packets and write to file
        """
        new_frame = av.VideoFrame(
            width=frame.width, height=frame.height, format=frame.format.name)
        for i in range(len(frame.planes)):
            new_frame.planes[i].update(frame.planes[i])
        pkt = None
        try:
            pkt = self.out_stream.encode(new_frame)
        except IOError as err:
            print("encoding failed: {0}".format(err))
        if pkt is not None:
            try:
                self.out_file.mux(pkt)
            except IOError:
                print('mux failed: ' + str(pkt))

    def take_picture(self, key_up):
        """Tell drone to take picture, image sent to file handler"""
        if key_up == 0:
            return
        self.drone.take_picture()

    def palm_land(self, key_up):
        """Tell drone to land"""
        if key_up == 0:
            return
        self.drone.palm_land()

    def toggle_zoom(self, key_up):
        """
        In "video" mode the self.drone sends 1280x720 frames.
        In "photo" mode it sends 2592x1936 (952x720) frames.
        The video will always be centered in the window.
        In photo mode, if we keep the window at 1280x720 that gives us ~160px on
        each side for status information, which is ample.
        Video mode is harder because then we need to abandon the 16:9 display size
        if we want to put the HUD next to the video.
        """
        if key_up == 0:
            return
        # Re-init the mission controls
        self.flags.mission_controls_init = False

        self.drone.set_video_mode(not self.drone.zoom)

    def toggle_mission(self, key_up):
        if key_up == 0:
            return
        self.flags.on_mission = not self.flags.on_mission
        # Always reset PID if mission flag's toggled
        self.reset_all_pids()
        print("on_mission:", self.flags.on_mission)
        return

    def flight_data_handler(self, event, sender, data):
        """Listener to flight data from the drone."""
        text = str(data)
        if self.prev_flight_data != text:
            self.prev_flight_data = text

    def handle_flight_received(self, event, sender, data):
        """Create a file in ~/Pictures/ to receive image from the drone"""
        path = '%s/Pictures/tello-%s.jpeg' % (
            os.getenv('HOME'),
            datetime.datetime.now().strftime(self.date_fmt))
        with open(path, 'wb') as out_file:
            out_file.write(data)
        print('Saved photo to %s' % path)

    def mission_control(self):
        distance = self.tracker.dists[self.use_w]
        box = self.tracker.target_box
        for i, var in enumerate(self.variables):
            if i < 2:
                center = box[i]+box[i+2]/2
                self.process_mv[var](
                    self.process_pv[var](
                        center if abs(center-self.process_pv[var].setpoint) > self.tolerances[var] \
                               else self.process_pv[var].setpoint
                    )
                )
            else:
                delta_distance = abs(distance-self.process_pv[var].setpoint)
                self.process_mv[var](
                    self.process_pv[var](
                        distance if  delta_distance > self.tolerances[var] \
                               else self.process_pv[var].setpoint
                    )
                )
                if delta_distance < self.tolerances[var]:
                    self.flags.on_tracking = not self.flags.on_tracking

        return

    def reset_all_pids(self):
        for var in self.variables:
            self.process_pv[var].reset()
        return

    def move_drone(self):
        if self.flags.fails_counter < 4:
            self.drone.up(self.speed)
        elif self.flags.fails_counter >= 4 and self.flags.fails_counter < 9:
            self.drone.clockwise(self.speed)
        elif self.flags.fails_counter >= 9 and self.flags.fails_counter < 14:
            self.drone.down(self.speed)
        else:
            self.flags.fails_counter = 0
        return


if __name__ == '__main__':
    main()
