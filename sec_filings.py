'''
This python file would have the following components in the pipeline

First iteration:
1. Get the data from sec filings based on the ticker(s) and year(s)
2. Extract texts for each of the section and store it as a separate metadata along with the section name and year 

Attributes: 1. An async function to call multiple threads to load the data (let's worry about it later)
2. Process each data and extract the texts from each section. 
'''

from typing import List
import asyncio
import aiohttp
from collections import defaultdict
from sec_edgar_downloader._utils import get_filing_urls_to_download
from prepline_sec_filings.sections import section_string_to_enum, validate_section_names, SECSection
from prepline_sec_filings.sec_document import SECDocument, REPORT_TYPES, VALID_FILING_TYPES

from prepline_sec_filings.fetch import (
    get_form_by_ticker, open_form_by_ticker, get_filing
)
import concurrent.futures 
import time 
from datetime import date
from enum import Enum
import re
import signal
import requests
from typing import Union,Optional
from ratelimit import limits, sleep_and_retry
import os
from unstructured.staging.base import convert_to_isd
from prepline_sec_filings.sections import (
    ALL_SECTIONS,
    SECTIONS_10K,
    SECTIONS_10Q,
    SECTIONS_S1,
)
import json
import argparse

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        except ValueError:
            pass
    def __exit__(self, type, value, traceback):
        try:
            signal.alarm(0)
        except ValueError:
            pass

# pipeline-api
def get_regex_enum(section_regex):
    class CustomSECSection(Enum):
        CUSTOM = re.compile(section_regex)
        
        @property
        def pattern(self):
            return self.value
            
    return CustomSECSection.CUSTOM
DATE_FORMAT_TOKENS = "%Y-%m-%d"
DEFAULT_BEFORE_DATE = date.today().strftime(DATE_FORMAT_TOKENS)
DEFAULT_AFTER_DATE = date(2000, 1, 1).strftime(DATE_FORMAT_TOKENS)

class SECExtractor:
    def __init__(self,tickers:List[str],amount:int,form_type:str,start_date:str=DEFAULT_AFTER_DATE,end_date:str=DEFAULT_BEFORE_DATE,sections:List[str]=['_ALL']):
        self.tickers = tickers
        self.amount = amount
        self.form_type = form_type
        self.start_date = start_date
        self.end_date = end_date
        self.sections = sections
    
    def get_accession_numbers(self,tic:str):
        final_dict = {}
        filing_metadata = get_filing_urls_to_download(
                    self.form_type,
                    tic,
                    self.amount,
                    self.start_date,
                    self.end_date,
                    include_amends=False
                )
        # fm.append(filing_metadata)
        acc_nums_yrs = [[fm.accession_number.replace("-",""),"20"+fm.accession_number.split("-")[1],fm.full_submission_url] for fm in filing_metadata]
        for acy in acc_nums_yrs:
            if tic not in final_dict:final_dict.update({tic:[]})
            final_dict[tic].append({"year":acy[1],"accession_number":acy[0],"url":acy[2]})
        return final_dict
    def get_all_text(self,section,all_narratives):
        all_texts = []
        for text_dict in all_narratives[section]:
            for key,val in text_dict.items():
                if key=="text":
                    all_texts.append(val)
        return ' '.join(all_texts)
    
    def get_text_from_acc_num(self,url:str):
        text = self.get_filing(
                    url, 
                    company='Unstructured Technologies', 
                    email='support@unstructured.io')
        all_narratives,filing_type = self.pipeline_api(text, m_section=self.sections)
        all_narrative_dict = dict.fromkeys(all_narratives.keys())

        for section in all_narratives:
            all_narrative_dict[section] = self.get_all_text(section,all_narratives)

        return all_narrative_dict,filing_type
    
    def pipeline_api(self,text, m_section=[], m_section_regex=[]):
        validate_section_names(m_section)

        sec_document = SECDocument.from_string(text)
        if sec_document.filing_type not in VALID_FILING_TYPES:
            raise ValueError(
                f"SEC document filing type {sec_document.filing_type} is not supported, "
                f"must be one of {','.join(VALID_FILING_TYPES)}"
            )
        results = {}
        if m_section == [ALL_SECTIONS]:
            filing_type = sec_document.filing_type
            if filing_type in REPORT_TYPES:
                if filing_type.startswith("10-K"):
                    m_section = [enum.name for enum in SECTIONS_10K]
                elif filing_type.startswith("10-Q"):
                    m_section = [enum.name for enum in SECTIONS_10Q]
                else:
                    raise ValueError(f"Invalid report type: {filing_type}")

            else:
                m_section = [enum.name for enum in SECTIONS_S1]
        for section in m_section:
            results[section] = sec_document.get_section_narrative(
                section_string_to_enum[section])
        
        for i,section_regex in enumerate(m_section_regex):
            regex_num = get_regex_enum(section_regex)
            with timeout(seconds=5):
                section_elements = sec_document.get_section_narrative(regex_num)
                results[f"REGEX_{i}"] = section_elements
        return {section:convert_to_isd(section_narrative) for section,section_narrative in results.items()},sec_document.filing_type
    @sleep_and_retry
    @limits(calls=10, period=1)
    def get_filing(self,
        url:str,
        company: str, email: str
    ) -> str:
        """Fetches the specified filing from the SEC EDGAR Archives. Conforms to the rate
        limits specified on the SEC website.
        ref: https://www.sec.gov/os/accessing-edgar-data"""
        session = self._get_session(company,email)
        response = session.get(url)
        response.raise_for_status()
        return response.text


    def _get_session(self,company: Optional[str] = None, email: Optional[str] = None) -> requests.Session:
        """Creates a requests sessions with the appropriate headers set. If these headers are not
        set, SEC will reject your request.
        ref: https://www.sec.gov/os/accessing-edgar-data"""
        if company is None:
            company = os.environ.get("SEC_API_ORGANIZATION")
        if email is None:
            email = os.environ.get("SEC_API_EMAIL")
        assert company
        assert email
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": f"{company} {email}",
                "Content-Type": "text/html",
            }
        )
        return session
    


