FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv


COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py","./"]
RUN mkdir output
ENTRYPOINT [ "python","starter.py", "2022","4" ]
