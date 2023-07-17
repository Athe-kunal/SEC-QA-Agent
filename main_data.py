from sec_filings import SECExtractor
import time
import json
import concurrent.futures
from collections import defaultdict
import os
tickers = ['TSLA','AAPL']
amount = 2
document_type = "10-K"
se = SECExtractor(tickers,amount,document_type,end_date="2022-12-31")
#290 seconds
os.makedirs("data",exist_ok=True)
def multiprocess_run(tic):
    # print(f"Started for {tic}")
    tic_dict = se.get_accession_numbers(tic)
    text_dict = defaultdict(list)
    for tic,fields in tic_dict.items():
        os.makedirs(f"data/{tic}",exist_ok=True)
        print(f"Started for {tic}")

        field_urls = [field['url'] for field in fields]
        years = [field['year'] for field in fields]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(se.get_text_from_acc_num,field_urls)
        for idx,res in enumerate(results):
            all_text,filing_type = res
            text_dict[tic].append({"year":years[idx],"ticker":tic,"all_texts":all_text,"filing_type":filing_type})
    return text_dict

if __name__ =="__main__":           
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(multiprocess_run,tickers)

    # final_dict = []
    for res in results:
        # final_dict.append(res)
        curr_tic = list(res.keys())[0]
        for data in res[curr_tic]:
            # print(data)
            curr_year = data["year"]
            os.makedirs(f"data/{curr_tic}/{curr_year}",exist_ok=True)
            curr_filing_type = data['filing_type']
            with open(f"data/{curr_tic}/{curr_year}/{curr_filing_type}.json","w") as f:
                json.dump(data,f,indent=4)
    # json.dump(final_dict, open("section_text_full.json", 'a') )
    # print(final_dict)
    print(f"It took {round(time.time()-start,2)} seconds")