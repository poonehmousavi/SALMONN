import pickle
import numpy as np
import matplotlib.pyplot as plt

dt = pickle.load(open('qwen_summarized.pkl', 'rb'))


all_dtws = []
all_l1s = []
all_l1s_v2 = []
all_l1s_v3 = []
all_l1paths = []

for key in dt.keys():
    all_dtws.append(dt[key]['dtw_errors'])
    all_l1s.append(dt[key]['l1_errors'])
    all_l1s_v2.append(dt[key]['l1_errors_v2'])
    all_l1s_v3.append(dt[key]['l1_errors_v3'])
    all_l1paths.append(dt[key]['l1_path_errors'])


all_dtws = np.array(all_dtws)
all_l1s = np.array(all_l1s)
all_l1s_v2 = np.array(all_l1s_v2)
all_l1s_v3 = np.array(all_l1s_v3)
all_l1paths = np.array(all_l1paths)

plt.plot(all_l1s.mean(0), label='l1')
#plt.plot(all_l1s_v2.mean(0), label='l1_v2')
#plt.plot(all_l1s_v3.mean(0), label='l1_v3')
#plt.plot(all_l1paths.mean(0), label='l1path')


#plt.plot(all_dtws.mean(0), label='dtw')

plt.legend()

plt.savefig('qwen_summarized.png')

