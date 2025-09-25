import os
import random

from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.trainer.trainer import Trainer
from chgnet.model import CHGNet
import torch
import pickle

torch.manual_seed(0)

def read_database(file_path):
    print("Reading databse...")
    if os.path.isfile(file_path):
        db = read(file_path,':')
        return db

    else:
        print("File not found! Exiting...")
        exit()

def get_values_from_db(db: list[Atoms]):
    """
    Extracts the magmoms, energy and force, stress from the dataset
    """

    structures = []
    magmoms = []
    energies_per_atom = []
    forces = []
    stress = []
    for atoms in db:
        # a try and except is used to ignore the atoms that don't have magnetic moments
        magmoms.append(atoms.arrays['initial_magmoms'])
        energies_per_atom.append(atoms.info['energy_per_atom'])
        forces.append(atoms.get_forces())
        structures.append(AseAtomsAdaptor.get_structure(atoms))
        stress.append(atoms.get_stress(voigt=False))

    return magmoms, energies_per_atom, forces, structures, stress


def train_chgnet(output_dir='output/models', epochs=1000, batch_size=40, learning_rate=1e-4, decay_fraction=1e-2):
    db_path = Path("../dataset/combinex.xyz")
    db = read_database(db_path)

    #shuffle the list
    random.seed(4)
    random.shuffle(db)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    magmoms, energies_per_atom, forces, structures, stress = get_values_from_db(db)

    # stress is not used
    print("Len of dataset:",len(structures))
    dataset = StructureData(
        structures=structures,
        energies=energies_per_atom,
        forces=forces,
        magmoms=magmoms)

    # load the trainer
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=batch_size, train_ratio=0.9, val_ratio=0.05)

    # save train val test loader as dictionay
    dct = {'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader}

    print("Saving train, val, test loaders to pickle file...")
    with open(f'{output_dir}/train_val_test.pkl', 'wb') as f:
        pickle.dump(dct, f)

    chgnet = CHGNet.load(model_name='0.2.0')

    trainer = Trainer(
                model=chgnet,
                targets="efm",
                scheduler = "CosineAnnealingLR",
                optimizer="Adam",
                criterion="Huber",
                learning_rate=learning_rate, 
                epochs=epochs,
                use_device=device,
                starting_epoch = 0,
                scheduler_params ={"decay_fraction": decay_fraction},
                )

    for param in chgnet.parameters():
        param.requires_grad = True

    # freezing the atom embedding layer
    # for param in chgnet.atom_embedding.parameters():
    #     param.requires_grad = False
    # Other layers to be frozen can be similarly added


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer.train(train_loader, 
                  val_loader, 
                  test_loader=test_loader, 
                  train_composition_model=True,
                  save_dir=output_dir)

if __name__ == "__main__":
    train_chgnet(epochs=50, batch_size=40, learning_rate=1e-4, decay_fraction=1e-2)
