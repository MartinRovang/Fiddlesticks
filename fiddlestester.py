#%%

import base64
import requests
import json
import numpy as np
import matplotlib.pyplot as plt



url = 'http://127.0.0.1:8000/'
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

body = {'data': data_base64}
headers = {'content-type': 'application/json'}

r = requests.post(url_make_model, data=json.dumps(body))#, headers=headers)


print(r.text)
# %%

url_check = url + 'check_drift'
data_not_equal = np.random.normal(4, 3, (1, 250, 250)).flatten()
data_ = np.random.choice(data_not_equal, size = int(np.sqrt(len(data_not_equal))))
data_base64_not_equal = base64.b64encode(data_not_equal.tobytes()).decode('utf-8')
body = {'data': data_base64_not_equal, 'projectid': '555be102-0760-4137-920b-3c19bf23a9ad'}
r = requests.post(url_check, data=json.dumps(body))


print(r.text)
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
