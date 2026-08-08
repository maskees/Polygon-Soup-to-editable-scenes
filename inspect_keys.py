import torch
from omegaconf import OmegaConf
import sys
sys.path.append(r"d:\3d model\external\TripoSR")
from tsr.system import TSR
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(repo_id="stabilityai/TripoSR", filename="config.yaml")
weight_path = hf_hub_download(repo_id="stabilityai/TripoSR", filename="model.ckpt")

cfg = OmegaConf.load(config_path)
OmegaConf.resolve(cfg)
model = TSR(cfg)

model_keys = set(model.state_dict().keys())
ckpt = torch.load(weight_path, map_location="cpu")
new_ckpt = {}
for k, v in ckpt.items():
    if k.startswith("image_tokenizer.model.encoder.layer."):
        new_k = k.replace("encoder.layer.", "layers.")
        new_k = new_k.replace(".attention.attention.query.", ".attention.q_proj.")
        new_k = new_k.replace(".attention.attention.key.", ".attention.k_proj.")
        new_k = new_k.replace(".attention.attention.value.", ".attention.v_proj.")
        new_k = new_k.replace(".attention.output.dense.", ".attention.o_proj.")
        new_k = new_k.replace(".intermediate.dense.", ".mlp.fc1.")
        new_k = new_k.replace(".output.dense.", ".mlp.fc2.")
        new_ckpt[new_k] = v
    else:
        new_ckpt[k] = v

ckpt_keys = set(new_ckpt.keys())

missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print("Missing keys from state_dict (model expects these but they are absent):")
for k in sorted(list(missing))[:20]:
    print("  ", k)

print("\nUnexpected keys in state_dict (ckpt has these but model doesn't expect them):")
for k in sorted(list(unexpected))[:20]:
    print("  ", k)
