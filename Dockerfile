FROM python:3.11
EXPOSE 8080
WORKDIR /app

COPY . /app/
RUN RUN pip install -r requirements.txt

ENTRYPOINT['streamlit','run,'Breast_Cancer_Analysis_App.py','server.port=8080','server.address=0.0.0.0']
