from __future__ import annotations

import os
import shutil
import warnings
warnings.simplefilter("ignore")
from functools import partial

import lightning as pl
import numpy as np
from pymatgen.io.vasp import Vasprun
from dgl.data.utils import split_dataset
from lightning.pytorch.loggers import CSVLogger

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
from matgl.utils.training import PotentialLightningModule

base_path = "/home/u9231474/matgl_chgent_ft"
model_save_path = os.path.join(base_path, "LATP_LCO_finetuned_model")
pre_trained_model = "CHGNet-MPtrj-2023.12.1-2.7M-PES"
dataset_path = os.path.join(base_path, "LATP_LCO_dataset")
folder_names = os.listdir(dataset_path)

if os.path.isdir("./MGLDataset"):
    shutil.rmtree("./MGLDataset")
    print("[CACHE] Removed ./MGLDataset")

structure_list = []
energy_list = []
force_list = []
stress_list = []
magmom_list = []

for folder in folder_names:
    vasp_dir = os.path.join(dataset_path, folder)
    vasprun_path = os.path.join(vasp_dir, "vasprun.xml")
    outcar_path = os.path.join(vasp_dir, "OUTCAR")
    
    if not os.path.isdir(vasp_dir):
        continue 

    try:
        vr = Vasprun(vasprun_path, parse_potcar_file=False)

        structure = vr.final_structure
        energy = float(vr.final_energy)
        force = np.array(vr.ionic_steps[-1]["forces"], dtype=float)   # (N,3)
        stress = np.array(vr.ionic_steps[-1]["stress"], dtype=float)  # (3,3)

        structure_list.append(structure)
        energy_list.append(energy)
        force_list.append(force)
        stress_list.append(stress)

        print(f"[OK] Loaded {vasprun_path}")

    except FileNotFoundError:
        print(f"[SKIP] Not found: {vasprun_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load {vasprun_path}: {e}")

print(f"\nKept {len(structure_list)} samples (after magmom filtering).")

labels = {
    "energies": [float(e) for e in energy_list],
    "forces":   [f.tolist() for f in force_list],
    "stresses": [s.tolist() for s in stress_list],
}

element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

dataset = MGLDataset(
    threebody_cutoff=3.0,
    structures=structure_list,
    converter=converter,
    labels=labels,
    include_line_graph=True,
)

train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)

my_collate_fn = partial(
    collate_fn_pes,
    include_line_graph=True,
    include_stress=True,
)


train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=4,
    num_workers=0,
)


chgnet_nnp = matgl.load_model(pre_trained_model)
model_pretrained = chgnet_nnp.model
property_offset = chgnet_nnp.element_refs.property_offset
lit_module_finetune = PotentialLightningModule(
    model=model_pretrained, element_refs=property_offset, lr=1e-3, include_line_graph=True, stress_weight=0.5
)
logger = CSVLogger("logs", name="CHGNet_fine_tuning")

trainer = pl.Trainer(max_epochs=5, accelerator="cuda", logger=logger, inference_mode=False)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)


# Save json 
from dump_parity import dump_parity

dump_parity(
    lit_module_finetune=lit_module_finetune,
    test_loader=test_loader,
    device="cuda",
    outfile="parity_test.json",
)



lit_module_finetune.model.save(model_save_path)
trained_model = matgl.load_model(path=model_save_path)

for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass
