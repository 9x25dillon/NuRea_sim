import random

def chaotic_flip_strength(value, token_entropy):
    strength = int(255 * max(0, 1 - token_entropy / 1024))  # example scaling
    if isinstance(value, int):
        return value ^ random.randint(1, max(1, strength))
    elif isinstance(value, str):
        n = max(1, int(len(value) * strength / 255))
        if n >= len(value):
            n = len(value)
        shuffled_part = ''.join(random.sample(value[:n], n))
        return shuffled_part + value[n:]
    return value

def increase_entropy(value, token_entropy=None):
    if isinstance(value, int):
        return value ^ random.randint(1000, 10000)
    elif isinstance(value, str):
        return ''.join(random.sample(value * 2, len(value)))
    return value

def decrease_entropy(value, token_entropy=None):
    if isinstance(value, int):
        return value & 0b1111
    elif isinstance(value, str):
        return ''.join(sorted(set(value)))
    return value