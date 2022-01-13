__version__ = '0.1.0'
from fastapi import FastAPI, File, UploadFile, Header, Body, HTTPException
from fastapi.responses import FileResponse
import asyncio
import time
import threading
import pickle
import datetime as dt
from kornia import feature
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
    data: str


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
        _ = np.frombuffer(base64.b64decode(s), dtype=np.float64)
        return type(_) == type(np.array([]))
    except Exception:
        return False

app = FastAPI()

@app.get("/")
async def read_root():
    return {'Operational': 'True'}


@app.post("/create_new_model")
async def create_model(data: InitModel):
    datadict = dict(data)
    base64encoded_reference_data = datadict['data']
    is_it_base64 = isBase64(base64encoded_reference_data)
    if is_it_base64:
        projectid = uuid.uuid4()
        base64encoded_reference_data = pickle.loads(base64.b64decode(base64encoded_reference_data))
        to_database = {'projectid': projectid, 'reference_data': base64encoded_reference_data, 'date': dt.datetime.now().strftime("%Y-%m-%d")}
        with open('fiddlesticks/database/'+str(projectid)+'.pkl', 'wb') as f:
            pickle.dump(to_database, f)
        return {'Status': f'{projectid}'}
    else:
        raise HTTPException(status_code=404, detail="Data not base64 encoded or not numpy array")



@app.post("/check_drift")
async def check_drift(data: CheckDriftPost):
    datadict = dict(data)
    base64encoded_target_data = datadict['data']
    model_id = datadict['projectid']
    is_it_base64 = isBase64(base64encoded_target_data)
    response = {}
    if is_it_base64:
        reference_data_target = pickle.loads(base64.b64decode(base64encoded_target_data))
        if os.path.exists('fiddlesticks/database/'+model_id+'.pkl'):
            with open('fiddlesticks/database/'+model_id+'.pkl', 'rb') as f:
                model_data = pickle.load(f)
            for feature_train in model_data['reference_data'].columns:
                ref_feature = model_data['reference_data'][feature_train].values
                target_feature = reference_data_target[feature_train].values
                stat, p = stats.ks_2samp(ref_feature, target_feature)
                reject_response = p < 0.05
                response[feature_train] = {'Reject Null': f'{reject_response}', 'pvalue':f'{p:.3f}'}
            
            data_report = Dashboard(tabs=[DataDriftTab()])
            data_report.calculate(model_data['reference_data'], reference_data_target, column_mapping = None)
            data_report.save(f"reports/{model_id}.html")
            return response
        else:
            raise HTTPException(status_code=404, detail="Model id does not exist")
    else:
        raise HTTPException(status_code=404, detail="Data not base64 encoded or not numpy array")
