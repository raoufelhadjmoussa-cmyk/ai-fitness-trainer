class RepCounter:
    def __init__(self):
        self.state = "up"   # up or down
        self.count = 0
        self.last_angle = 0

    def update(self, angle, down_thresh, up_thresh):
        """Return (new_rep, current_count, state)"""
        new_rep = False
        if angle < down_thresh and self.state == "up":
            self.state = "down"
        elif angle > up_thresh and self.state == "down":
            self.state = "up"
            self.count += 1
            new_rep = True
        return new_rep, self.count, self.state