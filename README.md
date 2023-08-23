# SEC-QA-Agent

To isntall the dependencies
```
pip install -r requirements.txt
```
Please add your OpenAI API key to `.env` file (inside the `app` directory) before running the codes 

Please check the data folder to see the supported companies. If you want to download any other data, please change the `data.yaml` file to download the 10-K or 10-Q

```console
cd app
```
To build the vector store

```console
python3 build_database.py
```

To run the streamlit application, run 

```console
streamlit run streamlit.py
```

Or you can directly build the docker file by 

```console
docker build -t sec-10k-qa .
```

And then run the docker container by 

```console
docker run -p 8501:8501 sec-10k-qa
```

Sample Answer Screenshot

![Sample Answer](https://github.com/Athe-kunal/SEC-QA-Agent/blob/main/Sample%20Answer.png)

Demo GIF

![Alt Text](https://github.com/Athe-kunal/SEC-QA-Agent/blob/main/Demo.gif)


