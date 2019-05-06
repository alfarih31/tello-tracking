class Flags:
    def __init__(self):
        self.tracker_init = False
        self.record = False

        self.mission_controls_init = False
        self.keydown = False

        self.on_tracking = False
        self.on_mission = False
        self.fails_counter = 0
