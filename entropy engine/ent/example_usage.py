from entropy_engine import Token, EntropyNode, EntropyEngine
from thermostat import ThermostaticTransform
from transforms import chaotic_flip_strength, increase_entropy, decrease_entropy
from oscillation import EntropyOscillator

# Thermostatic node example
thermo = ThermostaticTransform(chaotic_flip_strength)
root = EntropyNode("root", lambda val, ent: thermo(val, ent))
child = EntropyNode("child", lambda val, ent: thermo(val, ent), entropy_limit=900)
root.add_child(child)

engine = EntropyEngine(root, max_depth=3)
token = Token("entropy")
engine.run(token)
print(token)
print("Trace:", engine.trace())
print("Graph:", engine.export_graph())

# Oscillation example
osc = EntropyOscillator(low_thresh=400, high_thresh=900)
def oscillating_transform(value, token_entropy):
    transform = osc.select_transform(
        entropy=token_entropy,
        low_transform=increase_entropy,
        high_transform=decrease_entropy
    )
    return transform(value, token_entropy)

osc_node = EntropyNode("osc-node", oscillating_transform)
osc_engine = EntropyEngine(osc_node)
osc_token = Token("oscillate")
osc_engine.run(osc_token)
print(osc_token)
print("Osc Trace:", osc_engine.trace())
print("Osc thresholds:", osc.thresholds())