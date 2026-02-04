import torch
from chgnet.trainer import Trainer

ckpt_path = "/data01/tian_02/ACE/fine_tune_spin/LATP_LCO/01-23-2026/bestE_epoch13_e5_f186_s177_m118.pth.tar"
pt_path = ckpt_path.replace(".pth.tar", ".pt")

trainer = Trainer.load(ckpt_path)
model = trainer.model

torch.save(model.state_dict(), pt_path)

print(f"DONE: {pt_path}")
