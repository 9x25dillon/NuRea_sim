class ThermostaticTransform:
    def __init__(self, base_transform, min_entropy=400, max_entropy=900):
        self.base_transform = base_transform
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def __call__(self, value, token_entropy):
        # Normalize entropy to a scale (0.0 - 1.0)
        norm = min(max((token_entropy - self.min_entropy) / (self.max_entropy - self.min_entropy), 0), 1)
        strength = int(255 * (1 - norm))  # stronger flips at low entropy
        return self.base_transform(value, strength)