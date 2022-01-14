def initmodel(path = r'C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\train'):
    import glob
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import pickle
    import base64
    import json
    import requests
    url = 'http://127.0.0.1:8080/'
    url_make_model = url + 'create_new_model'
    files_list = glob.glob(path + '/*')
    # sample N amount of files
    #files_list = np.random.choice(files_list, N, replace=False)
    data_filesmean = np.array([])
    data_filesstd = np.array([])
    for file in files_list:
        #print(file)
        try:
            tmp_data = nib.load(file+'/FLAIR.nii.gz').get_fdata().flatten()
            data_filesstd = np.append(data_filesstd, np.std(tmp_data))
            data_filesmean = np.append(data_filesmean, np.mean(tmp_data))
        except Exception as e:
            print(e)
    out = np.concatenate((data_filesmean[:,None], data_filesstd[:,None]), axis = 1)
    out = pd.DataFrame(out, columns=['mean', 'std'])
    out = pickle.dumps(out)
    data_base64 = base64.b64encode(out).decode('utf-8')
    body = {'data': data_base64}
    r = requests.post(url_make_model, data=json.dumps(body))
    
    return r.json()['Status']

def checkdrift(path = r'C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\test', modelid = 'None', N = 3):
    import glob
    import numpy as np
    import nibabel as nib
    import pickle
    import base64
    import json
    import requests
    import pandas as pd
    url = 'http://127.0.0.1:8080/'
    url_check = url + 'check_drift'
    files_list = glob.glob(path + '/*')
    # sample N amount of files
    files_list = np.random.choice(files_list, N, replace=False)
    data_filesmean = np.array([])
    data_filesstd = np.array([])
    for file in files_list:
        #print(file)
        try:
            tmp_data = nib.load(file+'/FLAIR.nii.gz').get_fdata().flatten()
            data_filesstd = np.append(data_filesstd, np.std(tmp_data))
            data_filesmean = np.append(data_filesmean, np.mean(tmp_data))
        except Exception as e:
            print(e)
    out = np.concatenate((data_filesmean[:,None], data_filesstd[:,None]), axis = 1)
    out = pd.DataFrame(out, columns=['mean', 'std'])
    out = pickle.dumps(out)
    data_base64 = base64.b64encode(out).decode('utf-8')
    body = {'data': data_base64, 'projectid': modelid}
    r = requests.post(url_check, data=json.dumps(body))
    
    return r.json()

# projectid = initmodel()
# print(projectid)
# print(checkdrift(modelid = projectid))
print(checkdrift(modelid = 'e7c8c6ae-4a06-4904-85c9-ae5f506659f9'))

# projectid = initmodel(path = r'/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/train')
# print(projectid)
# print(checkdrift(path = r'/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test', modelid = projectid))



# #%% 

# import socketio
# import numpy as np
# import json
# sio = socketio.Client()

# sio.connect('http://127.0.0.1:5000', wait = True)

# packet = json.dumps({'data': [1, 2, 3]})

# sio.emit('msg',  packet)
# # %%
