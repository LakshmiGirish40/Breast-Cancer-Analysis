FROMM python:3.11
EXPOSE 8080
WORKDIR/APP

COPY . . /
RUN pip install requirements.txt

ENTRYPOPINT['streamlit','run,'Breast_Cancer_Analysis_App.py','server.port=8080','server.address=0.0.0.0']
