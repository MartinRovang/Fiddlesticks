# 
FROM python:3.9

# 
WORKDIR /fiddlemain

# 
COPY ./requirements.txt /fiddlemain/requirements.txt
COPY ./fiddlesticks /fiddlemain/fiddlesticks

# 
RUN python -m pip install --upgrade pip
RUN pip install -r /fiddlemain/requirements.txt

#
# 
CMD ["uvicorn", "fiddlesticks:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

