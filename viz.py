import torch
from unet import UNet

# --- instantiate your model ---
frames_out = 4          # how many frames your UNet predicts
action_dim = 19         # action size
mode = "interpolate"    # or "predict"

model = UNet(frames=frames_out, action_dim=action_dim, mode=mode)
model.eval()
model.to("cpu")

# --- save the state dict (recommended way) ---
pth_path = "unet_with_actions.pth"
torch.save(model.state_dict(), pth_path)

print(f"Model weights saved to {pth_path}")

# --- example of how to load it back ---
# model = UNet(frames=frames_out, action_dim=action_dim, mode=mode)
# model.load_state_dict(torch.load(pth_path, map_location="cpu"))
# model.eval()
