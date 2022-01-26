FROM tiangolo/uwsgi-nginx-flask:python3.8


COPY ./driftfrontend /driftfrontend

RUN python -m pip install --upgrade pip
RUN pip install seaborn pandas matplotlib

CMD ["python3", "/driftfrontend/frontendserver.py", "--host", "0.0.0.0", "--port", "5000"]