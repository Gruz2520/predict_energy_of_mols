import argparse
import pickle as pk
import pandas as pd
from tqdm import tqdm
from pymatgen.core.structure import Structure, Molecule
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import os
import pickle
from tensorflow.keras.callbacks import TensorBoard

def get_args():
    parser = argparse.ArgumentParser(description="Обучение megnet на наших данных.")

    parser.add_argument('--epoch', type=int, default=10, help='Количество эпох обучения')
    parser.add_argument('--name', type=str, default='def_name', help='Название проекта')
    parser.add_argument('--batch_s', type=int, default=32, help='Размер батча')
    parser.add_argument('--data_size', type=float, default=1.0, help='Используемый объем датасета')
    args = parser.parse_args()

    epoch = args.epoch
    name = args.name
    batch_s = args.batch_s
    data_size = args.data_size

    return epoch, name, batch_s, data_size


def parse_data():
    data = pd.read_csv('data/data_all_new.csv')

    molecules = []
    targets = []

    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        molecules.append(Molecule.from_str(row['xyz'], 'xyz'))
        targets.append(row['U_0'] / len(molecules[-1]))
        
    return molecules, targets


def create_structure_dict(structures: list, targets: list):
    structure_dict = {}

    for structure in tqdm(zip(structures, targets), total=len(structures)):
        if structure[0].formula in structure_dict:
            structure_dict[structure[0].formula].append(structure)
        else:
            structure_dict[structure[0].formula] = [structure]
        
    return structure_dict


def train_test_split_for_structures(structure_dict: dict, test_size: float = 0.1):
    structures_train, structures_test = [], []
    targets_train, targets_test = [], []
    train_all, test_all = [], []

    for _, data in tqdm(structure_dict.items()):
        if len(data) < 2:
            train_all += data
        else:
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
            train_all += train_data
            test_all += test_data
    
    random.shuffle(train_all)
    random.shuffle(test_all)

    for t_data in train_all:
        structures_train.append(t_data[0])
        targets_train.append(t_data[1])
    
    for te_data in test_all:
        structures_test.append(te_data[0])
        targets_test.append(te_data[1])
    
    return structures_train, targets_train, structures_test, targets_test


def main(epoch, name, batch_s, data_size):
    molecules, targets = parse_data()
    structure_dict = create_structure_dict(molecules, targets)
    molecules_train, targets_train, molecules_val_test, targets_val_test = train_test_split_for_structures(structure_dict)
    molecules_train = molecules_train[:int(len(molecules_train)*data_size)]
    targets_train = targets_train[:int(len(targets_train)*data_size)]
    structure_dict_val_test = create_structure_dict(molecules_val_test, targets_val_test)
    molecules_val, targets_val, molecules_test, targets_test = train_test_split_for_structures(structure_dict_val_test, 0.5)
    
    nfeat_bond = 10
    r_cutoff = 5
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = 0.5
    graph_converter = CrystalGraph(cutoff=r_cutoff)
    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
    
    loss_values = []

    class LossHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss_values.append(logs.get('loss'))

    log_dir = f"logs/fit/{name}"  # Путь для сохранения журналов TensorBoard
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True, update_freq = 'epoch')

    history = LossHistory()

    model.train(train_structures=molecules_train,
                train_targets=targets_train,
                validation_structures=molecules_val,
                validation_targets=targets_val,
                epochs=epoch,
                dirname=f'callback/{name}',
                batch_size=batch_s,
                callbacks=[history, tensorboard_callback])
    model.save_model(f'models/{name}.hdf5')
    # Create the directory structure if it does not exist
    os.makedirs('logs/lists/', exist_ok=True)

    with open(f'logs/lists/{name}.pkl', 'wb') as f:
        pickle.dump(loss_values, f)
    

if __name__ == '__main__':
    epoch, name, batch_s, data_size = get_args()
    main(epoch, name, batch_s, data_size)
