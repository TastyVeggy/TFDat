from tfdat.tracker.norfair import NorFair


class Tracker:
    def __init__(self, detector: object) -> None:

        self.tracker = NorFair(detector)

    def detect_and_track(self, image, config: dict):
        return self.tracker.detect_and_track(image, config)
