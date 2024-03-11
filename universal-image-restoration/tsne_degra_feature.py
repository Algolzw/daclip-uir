import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

base_path = '/root/workplace/daclip-uir/degra_feature_data'
distortion = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']

#get all filename in the base_path
files = os.listdir(base_path)
deg_type_list = []
deg_feature_list = []

for file in tqdm(files):
    if file.endswith('.npy'):
        feature = np.load(os.path.join(base_path, file))
        deg_type = file.split('_')[0]
        deg_type_list.append(deg_type)
        deg_feature_list.append(feature)
deg_feature_list = np.array(deg_feature_list)
#Conduct T-SNE on deg_feature_list
tsne = TSNE(n_components=2, random_state=0)
actual = np.array(deg_type_list)
deg_feature_list_2d = deg_feature_list.reshape(deg_feature_list.shape[0], -1)
cluster = np.array(tsne.fit_transform(np.array(deg_feature_list_2d)))
print(cluster.shape)
plt.figure(figsize=(10, 10))
for i, label in zip(range(10), distortion):
    plt.scatter(cluster[actual == label, 0], cluster[actual == label, 1], label=label)
plt.legend()
plt.savefig('/root/workplace/daclip-uir/universal-image-restoration/degra_feature_tsne.png')

