import chgnet
from pymatgen.core import Structure
from chgnet.model import CHGNet
from parse_vasp_dir import parse_vasp_dir
import os
import torch 
from torch import Tensor

base_path = "/data01/tian_02/SSE_ML/LiLaTeO/fine_tune/700K"
folder_names = os.listdir(os.path.join(base_path, "dft_data"))
chgnet = CHGNet.load()

dataset_dict = {}

for folder in folder_names:
    dataset_dict_temp = parse_vasp_dir(file_root=os.path.join(base_path, "dft_data", folder))
    for key in dataset_dict_temp.keys():
        dataset_dict.setdefault(key, []).extend(dataset_dict_temp[key])

# Convert structure to CHGNet graph
from chgnet.graph import CrystalGraphConverter
from dataset import StructureData, get_train_val_test_loader
graph_name = []
converter = CrystalGraphConverter()
for idx, struct in enumerate(dataset_dict["structure"]):
    graph = converter(struct)
    graph_name.append(idx)
    torch.save(graph, os.path.join(base_path + "/pt", f"{idx}.pt"))

labels = {}
key = ['energy_per_atom', 'force', 'stress']
for i, mp_id in enumerate(graph_name, start=0):
    labels[mp_id] = {}
    new_dict = {}
    for k in key:
        new_dict[k] = dataset_dict[k][i]
    labels[mp_id][mp_id] = new_dict

from dataset import GraphData
graph_data = GraphData(
    graph_path=base_path + "/pt/",
    labels=labels
)

import trainer 
from trainer import Trainer
trainer = Trainer(
    model=chgnet,
    targets="efs",
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=1,
    learning_rate=1e-2,
    use_device="cuda",  # Change to "cuda" if using GPU
    print_freq=1,
)

train_loader, val_loader, test_loader = graph_data.get_train_val_test_loader(batch_size=5, train_ratio=0.7, val_ratio=0.1)

trainer.train(train_loader, val_loader, test_loader)

model = trainer.model
best_model = trainer.best_model  # Best model based on validation energy MAE

# For validation set
val_errors = trainer._validate(val_loader, is_test=False)

# For test set (and save results)
test_errors = trainer._validate(test_loader, is_test=True, test_result_save_path=base_path)
