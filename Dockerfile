FROM tensorflow/tensorflow:1.15.5-gpu

MAINTAINER Illia Herasymenko, illia.cgerasimenko@gmail.com

RUN mkdir -p /app

WORKDIR /app

COPY docker_requirements.txt .

RUN pip install --upgrade pip && /usr/bin/python3 -m pip install --upgrade pip && pip install -r docker_requirements.txt

COPY . .

CMD ["python", "flask_app/server.py"]
