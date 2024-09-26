import pickle as pk
import pandas as pd
import numpy as np
from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer
import tensorflow as tf

print('loading the MPF dataset 2021')
with open('path to block_0.p', 'rb') as f:
    data = pk.load(f)

with open('path to block_1.p', 'rb') as f:
    data2 = pk.load(f)
print('MPF dataset 2021 loaded')
data.update(data2)
df = pd.DataFrame.from_dict(data)

dataset_train = []
for idx, item in df.items():
    for iid in range(len(item['energy'])):
        dataset_train.append({"atoms":item['structure'][iid], "energy":item['energy'][iid] / len(item['force'][iid]), "force": np.array(item['force'][iid]), "stress": np.array(item['stress'][iid])})

print('using %d samples to train, '%(len(dataset_train)))


m3gnet = M3GNet(is_intensive=False)
potential = Potential(model=m3gnet)

trainer = PotentialTrainer(
    potential=potential, optimizer=tf.keras.optimizers.Adam(1e-3)
)
trainer.train(
    [dataset_train[0]['atoms']],
    [dataset_train[0]['energy']],
    [dataset_train[0]['force']],
    [dataset_train[0]['stress']],
    validation_graphs_or_structures=[dataset_train[0]['atoms']],
    val_energies=[dataset_train[0]['energy']],
    val_forces=[dataset_train[0]['force']],
    val_stresses=[dataset_train[0]['stress']],
    epochs=2,
    fit_per_element_offset=True,
    save_checkpoint=False,
)