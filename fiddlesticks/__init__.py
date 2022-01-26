__version__ = '0.1.0'
from pydoc import describe
from fastapi import FastAPI, File, UploadFile, Header, Body, HTTPException
from fastapi.responses import FileResponse
import asyncio
import time
import threading
import pickle
import datetime as dt
import numpy as np
import json
import os
from scipy import stats
from pydantic import BaseModel
from typing import Optional
import base64
import uuid
import pandas as pd

from evidently.dashboard import Dashboard
from evidently.tabs import (
    DataDriftTab
)

class InitModel(BaseModel):
    description: str

class InitRef(BaseModel):
    data: str
    projectid: str


class CheckDriftPost(BaseModel):
    data: str
    projectid: str



#return {'Reject Null': f'{reject_response}', 'pvalue':f'{p:.3f}', 'Null hypothesis': 'The null hypothesis is that the two distributions are identical, F(x)=G(x) for all x'}

def isBase64(s):
    try:
        return base64.b64encode(base64.b64decode(s)).decode('utf-8') == s
    except Exception:
        return False

def isNumpy(s):
    try:
        return type(s) == type(np.array([]))
    except Exception:
        return False

app = FastAPI()

@app.get("/")
async def read_root():
    return {'Operational': 'True'}


@app.post("/create_new_model")
async def create_model(input: InitModel):
    input = dict(input)
    description = input['description']
    projectid = uuid.uuid4()
    to_database = {'projectid': projectid, 'description': description, 'date': dt.datetime.now().strftime("%Y-%m-%d")}
    with open('fiddlesticks/database/'+str(projectid)+'.pkl', 'wb') as f:
        pickle.dump(to_database, f)
    return {'Status': f'{projectid}'}


@app.post("/add_reference_data")
async def create_model(data: InitRef):
    datadict = dict(data)
    base64encoded_reference_data = datadict['data']
    projectid = datadict['projectid']
    if os.path.exists('fiddlesticks/database/'+projectid+'.pkl'):
        is_it_base64 = isBase64(base64encoded_reference_data)
        if is_it_base64:
            base64encoded_reference_data = pickle.loads(base64.b64decode(base64encoded_reference_data))
            is_it_numpy = isNumpy(base64encoded_reference_data)
            if is_it_numpy:
                projectfile = pickle.load(open('fiddlesticks/database/'+projectid+'.pkl', 'rb'))
                # Flatten and normalize data
                base64encoded_reference_data = base64encoded_reference_data.flatten()
                base64encoded_reference_data_ent = base64encoded_reference_data - np.min(base64encoded_reference_data)
                entropy_value = stats.entropy(base64encoded_reference_data_ent)
                if 'reference_data' not in projectfile:
                    projectfile['reference_data'] = {'entropy': np.array([entropy_value]), 'mean': np.array([np.mean(base64encoded_reference_data)]), 'std': np.array([np.std(base64encoded_reference_data)])}
                else:
                    projectfile['reference_data']['entropy'] = np.append(projectfile['reference_data']['entropy'], entropy_value)
                    projectfile['reference_data']['mean'] = np.append(projectfile['reference_data']['mean'], np.mean(base64encoded_reference_data))
                    projectfile['reference_data']['std'] = np.append(projectfile['reference_data']['std'], np.std(base64encoded_reference_data))
                with open('fiddlesticks/database/'+str(projectid)+'.pkl', 'wb') as f:
                    pickle.dump(projectfile, f)
                return {'Status': f'Added ref data to: {projectid}, entropy: {entropy_value}'}
            else:
                raise HTTPException(status_code=404, detail="Data not numpy array!")
        else:
            raise HTTPException(status_code=404, detail="Data not base64 encoded!")
    else:
        raise HTTPException(status_code=404, detail="Project not found, use correct project id")


@app.post("/check_drift")
async def check_drift(data: CheckDriftPost):
    datadict = dict(data)
    base64encoded_target_data = datadict['data']
    model_id = datadict['projectid']
    is_it_base64 = isBase64(base64encoded_target_data)
    response = {}
    if is_it_base64:
        reference_data_target = pickle.loads(base64.b64decode(base64encoded_target_data))
        is_it_numpy = isNumpy(reference_data_target)
        if is_it_numpy:
            if os.path.exists('fiddlesticks/database/'+model_id+'.pkl'):
                reference_data_target = reference_data_target.flatten()
                reference_data_target_ent = reference_data_target - np.min(reference_data_target)
                reference_data_target_ent = np.array([stats.entropy(reference_data_target_ent)])

                reference_data_target = {'entropy': reference_data_target_ent, 'mean': np.array([np.mean(reference_data_target)]), 'std': np.array([np.std(reference_data_target)])}
                print(reference_data_target)
                with open('fiddlesticks/database/'+model_id+'.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                for feature_train in model_data['reference_data']:
                    ref_feature = model_data['reference_data'][feature_train]
                    target_feature = reference_data_target[feature_train]
                    stat, p = stats.ks_2samp(ref_feature, target_feature)
                    reject_response = p < 0.05
                    response[feature_train] = {'Reject Null': f'{reject_response}', 'pvalue':f'{p:.3f}'}

                reference_data = pd.DataFrame(model_data['reference_data'], columns= list(reference_data_target.keys()))
                reference_data_target = pd.DataFrame(reference_data_target, columns= list(reference_data_target.keys()))
                data_report = Dashboard(tabs=[DataDriftTab()])
                data_report.calculate(reference_data, reference_data_target, column_mapping = None)
                time_now = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                data_report.save(f"fiddlesticks/database/{model_id}_report_{time_now}.html")

                
                if os.path.exists('fiddlesticks/database/'+model_id+'_historical.pkl'):
                    with open('fiddlesticks/database/'+model_id+'_historical.pkl', 'rb') as f:
                        historical_data = pickle.load(f)
                    historical_data[time_now] = {}
                    for feature_train in response:
                        historical_data[time_now][feature_train] = {'Drift detected': f'{response[feature_train]["Reject Null"]}', 'pvalue':f'{response[feature_train]["pvalue"]}'}
                        historical_data[time_now]['data'] = reference_data_target
                    
                    with open('fiddlesticks/database/'+model_id+'_historical.pkl', 'wb') as f:
                        pickle.dump(historical_data, f)
                else:
                    historical_data = {}
                    historical_data[time_now] = {}
                    for feature_train in response:
                        historical_data[time_now][feature_train] = {'Drift detected': f'{response[feature_train]["Reject Null"]}', 'pvalue':f'{response[feature_train]["pvalue"]}'}
                        historical_data[time_now]['data'] = reference_data_target
                    
                    with open('fiddlesticks/database/'+model_id+'_historical.pkl', 'wb') as f:
                        pickle.dump(historical_data, f)
                return response
            else:
                raise HTTPException(status_code=404, detail="Model id does not exist")
        else:
            raise HTTPException(status_code=404, detail="Data not numpy array!")
    else:
        raise HTTPException(status_code=404, detail="Data not base64 encoded!")
