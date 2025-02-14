import matplotlib.pyplot as plt
import pickle
import torch

data = pickle.load(open('qwen_summarized.pkl', 'rb'))

keys = list(data.keys())
N = 9

layer = 32 

for i, key in enumerate(keys[:N]): 
    plt.subplot(3, 3, i+1)
    plt.imshow(data[key]['innerprods'][layer].transpose(1, 0))
    plt.xlabel('Audio token time steps')
    plt.ylabel('Text token time steps')

plt.show()




