FROM tiangolo/uwsgi-nginx-flask:python3.8


COPY ./driftfrontend /driftfrontend

CMD ["python3", "/driftfrontend/frontendserver.py", "--host", "0.0.0.0", "--port", "5000"]