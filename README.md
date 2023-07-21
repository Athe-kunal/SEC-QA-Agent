# SEC-QA-Agent

To isntall the dependencies
```
pip install -r requirements.txt
```
Please add your OpenAI API key to `.env` file before running the codes 

Please check the data folder to see the supported companies. If you want to download any other data, please change the `data.yaml` file to download the 10-K or 10-Q

To build the vector store
```
python3 build_database.py
```

To run the streamlit application, run 

```
streamlit run streamlit.py
```

Sample Answer Screenshot

![Sample Answer](https://github.com/Athe-kunal/SEC-QA-Agent/blob/main/Sample%20Answer.png)


