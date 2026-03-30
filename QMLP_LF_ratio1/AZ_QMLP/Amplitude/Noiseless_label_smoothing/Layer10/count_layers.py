import torch

state_dict = torch.load("qmlp-az-amplitude-label-flip-run_best_run.pt", map_location="cpu")

# Count unique rotation layers
rot_layers = set(k for k in state_dict.keys() if "rot_layer" in k)
crx_layers = set(k for k in state_dict.keys() if "crx_layer" in k)

# Extract layer numbers
rot_nums = set(int(k.split("rot_layer_")[1].split(".")[0]) for k in rot_layers)
crx_nums = set(int(k.split("crx_layer_")[1].split(".")[0]) for k in crx_layers)

print("Number of rotation layers:", len(rot_nums))
print("Number of CRX layers:", len(crx_nums))
