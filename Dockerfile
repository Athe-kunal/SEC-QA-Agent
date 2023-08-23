FROM python:3.10-slim

COPY ./requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

WORKDIR /app
COPY . /app
EXPOSE 8051

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "streamlit.py", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]