from types import SimpleNamespace

TEAM = None
PROJECT = "llmapps"
JOB_TYPE = "SEC-Project"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="athe_kunal/llmapps/vector_store:latest",
    chat_prompt_artifact="athe_kunal/llmapps/chat_prompt:latest",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-3.5-turbo"
)

#LLM 1
DEFINITIONS_10Q = '''

FINANCIAL_STATEMENTS: This section in the 10-Q document is a comprehensive report of a company's financial performance during the quarter. It includes the income statement, balance sheet, and cash flow statement. The financial statements section provides detailed information on the company's revenues, expenses, assets, liabilities, and cash flows.

MANAGEMENT_DISCLOSURE: This part of the 10-Q form is essentially a narrative explanation, provided directly by management, of how the company performed during the quarter, the company's current financial condition, and management's perspective on future performance. The discussion is intended to provide context and insight beyond what the financial data alone can show.

MARKET_RISK_DISCLOSURES: This section reveals a company's exposure to potential financial losses that could occur due to market changes such as fluctuations in interest rates, foreign exchange rates, commodity prices, or equity prices. It also includes the strategies or measures taken by the company to manage or mitigate these risks.

CONTROLS_AND_PROCEDURES: In this part of the document, the company outlines the effectiveness and any changes in its internal control over financial reporting and disclosure controls and procedures. This helps investors assess the quality of the company's financial reporting and its ability to prevent fraud.

LEGAL_PROCEEDINGS: Here, the company discloses any material pending legal proceedings, other than ordinary routine litigation incidental to the business. This can include significant litigation, governmental inquiries, or regulatory challenges that the company is facing, which may impact its operations or financial health.

RISK_FACTORS: This section enumerates the significant risks that could adversely affect the company's business, operations, industry, financial position, or future financial performance. These risks can range from operational risks to financial, regulatory, and strategic risks.

USE_OF_PROCEEDS: This section is particularly relevant for newly public companies or companies that recently issued debt or equity securities. It provides information on how the company has used or plans to use the funds raised from these issues, including details on any material changes in the planned use of proceeds from what the company originally disclosed in the offering.

DEFAULTS: This section outlines any significant defaults on senior securities, such as bonds, notes, or preferred stock, that could have an adverse effect on the company's financial position. This information is critical for bondholders and preferred stockholders as it impacts their risk and return.

MINE_SAFETY: For companies engaged in mining operations, this section discloses any significant mine safety violations or other regulatory matters required by the Mine Safety and Health Administration. These disclosures provide insights into the company's compliance with safety regulations and potential liabilities.

OTHER_INFORMATION: This section serves as a catch-all for any material information that doesn't fit into the other sections. It could include a wide variety of information, such as disclosures about unregistered sales of equity securities, material changes in the rights of security holders, or changes in the company's certifying accountant.
'''

DEFINITIONS_10K = """
BUSINESS: Provides an overview of the company's operations, products, services, markets, and competitive landscape.

RISK_FACTORS: Identifies and discusses the potential risks and uncertainties that could affect the company's performance and future prospects.

UNRESOLVED_STAFF_COMMENTS: Addresses any outstanding comments or inquiries raised by the SEC staff during the review process.

PROPERTIES: Describes the company's owned or leased properties, including locations, facilities, and real estate holdings.

LEGAL_PROCEEDINGS: Discloses any ongoing or pending legal actions, disputes, or regulatory matters involving the company.

MINE_SAFETY: Pertains specifically to mining companies, providing information about safety and compliance measures in mining operations.

MARKET_FOR_REGISTRANT_COMMON_EQUITY: Discusses the company's stock market, trading volume, and other relevant information related to its common equity.

MANAGEMENT_DISCUSSION: Presents the company's management's analysis of its financial performance, results of operations, and future plans.

MARKET_RISK_DISCLOSURES: Highlights the potential risks and uncertainties arising from market conditions that could impact the company's financial position.

FINANCIAL_STATEMENTS: Provides the audited financial statements, including the balance sheet, income statement, cash flow statement, and accompanying footnotes.

ACCOUNTING_DISAGREEMENTS: Addresses any disagreements or reservations between the company and its auditors regarding accounting practices or financial reporting.

CONTROLS_AND_PROCEDURES: Describes the company's internal controls and procedures for financial reporting and compliance with regulations.

FOREIGN_JURISDICTIONS: Covers any significant operations or risks related to the company's activities in foreign countries.

MANAGEMENT: Provides information about the company's executive officers, their roles, responsibilities, and background.

COMPENSATION: Details the compensation packages, including salaries, bonuses, stock options, and other benefits, for the company's executives.

PRINCIPAL_STOCKHOLDERS: Lists the major shareholders or beneficial owners who hold a significant stake in the company.

RELATED_PARTY_TRANSACTIONS: Discloses any transactions or relationships between the company and its directors, officers, or affiliates.

ACCOUNTING_FEES: Reports the fees paid to the company's independent auditors for their services, including audit and non-audit services.

EXHIBITS: Includes various supporting documents, agreements, contracts, or other relevant materials referenced throughout the 10-K report.

FORM_SUMMARY: Provides a summary of the key information contained in the 10-K report, offering an overview of the company's financial performance and prospects
"""
NUM_SECTION_RETURN = 5


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

NUM_RESULTS = 50