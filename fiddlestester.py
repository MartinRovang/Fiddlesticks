#%%

import base64
import requests
import json
import numpy as np
import matplotlib.pyplot as plt



url = 'http://127.0.0.1:8080/'
x = requests.get(url)
print(x.text)

#%%

# Encode numpy array to base64
data_ = np.random.normal(4, 2, (100, 250, 250)).flatten()
data_ = np.random.choice(data_, size = int(np.sqrt(len(data_))))

plt.hist(data_, cumulative=True, density=True, histtype='step', bins=1000)
plt.show()
data_base64 = base64.b64encode(data_.tobytes()).decode('utf-8')

# decode numpy array
data_decoded = np.frombuffer(base64.b64decode(data_base64), dtype=np.float64)
print(data_decoded.shape)
#
# %%
url_make_model = url + 'create_new_model'

body = {'description': 'This is a test model'}
headers = {'content-type': 'application/json'}

r = requests.post(url_make_model, data = json.dumps(body))#, headers=headers)


print(r.text)

#%%
import pickle
for i in range(0, 100):
    url_check = url + 'add_reference_data'
    data_not_equal = np.random.normal(np.random.randint(1,10), np.random.randint(1,4), (1, 250, 250))
    out = pickle.dumps(data_not_equal)
    data_base64_not_equal = base64.b64encode(out).decode('utf-8')

    body = {'data': data_base64_not_equal, 'projectid': '024f2853-7a2c-4494-a369-805758bb2354'}
    r = requests.post(url_check, data=json.dumps(body))
    print(r.text)

# %%
from scipy import stats
import time

for i in range(0, 100):
    url_check = url + 'check_drift'
    data_base64_not_equal = np.random.normal(np.random.randint(1,21), np.random.randint(1,9), (1, 250, 250))
    out = pickle.dumps(data_base64_not_equal)
    out = base64.b64encode(out).decode('utf-8')

    body = {'data': out, 'projectid': '024f2853-7a2c-4494-a369-805758bb2354'}
    r = requests.post(url_check, data=json.dumps(body))
    print(r.text)
    time.sleep(5)
# %%
import requests
import json
url_test = url + 'test'

body = { "data": f"{data_base64_not_equal}" }
headers = {'content-type': 'application/json'}

r = requests.post(url_test, data=json.dumps(body))#, headers=headers)


print(r.text)
# %%


import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats

trian = np.random.normal(0, 1, (100, 250, 250)).flatten()
trian = np.random.choice(trian, size = int(np.sqrt(len(trian))))
newsample = np.random.normal(0, 1, (1, 250, 250)).flatten()
newsample = np.random.choice(newsample, size = int(np.sqrt(len(newsample))))
statistic, p = stats.ks_2samp(trian, newsample)
print(statistic, p)
# %%
print(int(np.sqrt(100*250*250)))

# %%

import seaborn as sns
entropy_array = []
mean_array = []
std_array = []
for i in range(0, 100):
    # test = np.random.normal(i, 2+i//3, (1, 250, 250)).flatten() + np.random.uniform(i, 1+i//3, (1, 250, 250)).flatten()
    test = np.random.uniform(1, 1+i//3, (1, 250, 250)).flatten()
    print(test)
    test_ent = test - np.min(test)
    test_ent = test_ent / np.max(test_ent)
    entropy_array.append(stats.entropy(test_ent))
    # mean_array.append(np.mean(test))
    # std_array.append(np.std(test))


data = {'entropy': entropy_array, 'mean': mean_array, 'std': std_array}


sns.kdeplot(data = data, shade=True)
plt.show()
    
# %%

test = np.random.normal(100, 5, 10000).flatten()
test = np.random.uniform(1, 2, 10000).flatten()
# test_ent = test - np.min(test)
# test_ent = test_ent / np.max(test_ent)
print(stats.entropy(test, base=2))

plt.hist(test)
plt.show()
# %%
