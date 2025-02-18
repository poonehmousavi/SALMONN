import pickle
import numpy as np
import matplotlib.pyplot as plt

dt = pickle.load(open('qwen_summarized.pkl', 'rb'))


all_dtws = []
all_l1s = []
for key in dt.keys():
    all_dtws.append(dt[key]['dtw_errors'])
    all_l1s.append(dt[key]['l1_errors'])

all_dtws = np.array(all_dtws)
all_l1s = np.array(all_l1s)

#plt.plot(all_l1s.mean(0), label='l1')
plt.plot(all_dtws.mean(0), label='dtw')

plt.legend()

plt.show()
import pdb; pdb.set_trace()

