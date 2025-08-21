class EntropyOscillator:
    def __init__(self, low_thresh=400, high_thresh=900, learning_rate=10):
        self.low = low_thresh
        self.high = high_thresh
        self.learning_rate = learning_rate
        self.last_entropy = None

    def adjust_thresholds(self, current_entropy):
        if self.last_entropy is not None:
            delta = current_entropy - self.last_entropy
            # If entropy increased too much, narrow thresholds
            if delta > 0:
                self.low += self.learning_rate
                self.high -= self.learning_rate
            else:
                self.low -= self.learning_rate
                self.high += self.learning_rate

            # Keep in bounds
            self.low = max(0, min(self.low, self.high - 50))
            self.high = min(1024, max(self.high, self.low + 50))
        self.last_entropy = current_entropy

    def select_transform(self, entropy, low_transform, high_transform, neutral_transform=None):
        self.adjust_thresholds(entropy)
        if entropy < self.low:
            return low_transform
        elif entropy > self.high:
            return high_transform
        return neutral_transform or (lambda v, e: v)  # No-op

    def thresholds(self):
        return (self.low, self.high)