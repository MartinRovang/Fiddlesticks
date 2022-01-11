__version__ = '0.1.0'
from fastapi import FastAPI, File, UploadFile, Header, Body
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
# class Item(BaseModel):
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None


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
async def create_model(data: dict = Body(...), header: Optional[str] = Header(None)):
    if 'data' in data:
        base64encoded_reference_data = data['data']
        is_it_base64 = isBase64(base64encoded_reference_data)
        is_it_numpy = isNumpy(base64encoded_reference_data)
        if is_it_base64 and is_it_numpy:
            projectid = uuid.uuid4()
            to_database = {'projectid': projectid, 'reference_data': base64encoded_reference_data, 'date': dt.datetime.now().strftime("%Y-%m-%d")}
            with open('fiddlesticks/database/'+str(projectid)+'.pkl', 'wb') as f:
                pickle.dump(to_database, f)
            return {'Status': f'{projectid}'}
        else:
            return {'Status': 'Error, data not base64 encoded or not numpy array'}
    else:
        return {'Status': 'must be on form {data: base64encoded_reference_data}'}

@app.post("/check_drift")
async def check_drift(data: dict = Body(...), header: Optional[str] = Header(None)):
    if 'data' in data and 'projectid' in data:
        base64encoded_target_data = data['data']
        model_id = data['projectid']
        is_it_base64 = isBase64(base64encoded_target_data)
        is_it_numpy = isNumpy(base64encoded_target_data)
        if is_it_base64 and is_it_numpy:
            reference_data_target = base64.b64decode(base64encoded_target_data)#.decode('utf-8')
            target_data_decoded = np.frombuffer(reference_data_target, dtype=np.float64)
            if os.path.exists('fiddlesticks/database/'+model_id+'.pkl'):
                with open('fiddlesticks/database/'+model_id+'.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    reference_data = base64.b64decode(model_data['reference_data'])#.decode('utf-8')
                    reference_data_decoded = np.frombuffer(reference_data, dtype=np.float64)

                    # hist, bin = np.histogram(reference_data_decoded, bins=len(), density=True)
                    # hist2, bin2 = np.histogram(target_data_decoded, bins=50, density=True)
                    stat, p = stats.ks_2samp(reference_data_decoded, target_data_decoded)
                    reject_response = p < 0.05
                    return {'Reject Null': f'{reject_response}', 'pvalue':f'{p:.3f}', 'Null hypothesis': 'The null hypothesis is that the two distributions are identical, F(x)=G(x) for all x'}
            else:
                return {'Status': 'Error, model id does not exist'}
        else:
            return {'Status': 'Error, data not base64 encoded or not numpy array'}
    else:
        return {'Status': 'must be on form {data: base64encoded_target_data, projectid: model_id}'}