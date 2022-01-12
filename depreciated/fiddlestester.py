#%%

import base64
import requests
import numpy as np


url = 'http://127.0.0.1:8000/'
x = requests.get(url)
print(x.text)

#%%

# Encode numpy array to base64
data_ = np.random.normal(4, 2, 10000)
data_base64 = base64.b64encode(data_.tobytes()).decode('utf-8')

# decode numpy array
data_decoded = np.frombuffer(base64.b64decode(data_base64), dtype=np.float64)
print(data_decoded)
#
# %%
import requests
import json
url_make_model = url + 'create_new_model'

body = {'data': data_base64}
headers = {'content-type': 'application/json'}

r = requests.post(url_make_model, data=json.dumps(body))#, headers=headers)


print(r.text)
# %%

url_check = url + 'check_drift'
data_not_equal = np.random.normal(4, 2, 100)
data_base64_not_equal = base64.b64encode(data_not_equal.tobytes()).decode('utf-8')
body = {'data': data_base64_not_equal, 'projectid': '95101578-83bf-4db5-a696-cdf071dd6d9d'}

r = requests.post(url_check, data=json.dumps(body))


print(r.text)
# %%
import requests
import json
url_test = url + 'test'

body = { "data": f"{data_base64}" }
headers = {'content-type': 'application/json'}

r = requests.post(url_test, data=json.dumps(body))#, headers=headers)


print(r.text)
# %%
