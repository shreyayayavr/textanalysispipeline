# BlackOffer Internship Assessment
This project is part of the BlackOffer Internship Assessment. It consists of two main Python scripts that:
1. **Extract Articles** from a list of URLs provided in an Excel file.
2. **Analyze Extracted Articles** using linguistic metrics, sentiment dictionaries, and stop words filtering.

**Article Extraction** (extract_articles.py)

Objective:
Extract clean text from web articles and save each article as a .txt file for further analysis.

Key Features:
Reads URLs and URL_IDs from an Excel file.

Fetches HTML using requests with retry and custom headers.

Cleans HTML by removing scripts,styles,navigation,headers,footers,and ads.

Detects the article title and main content container intelligently.

Extracts paragraphs and filters out irrelevant content using regex patterns.

Saves each article as a .txt file named after its URL_ID.

Logs extraction status and word counts in a CSV file.

**Article Analysis** (analyze_articles.py)

Objective:
Compute sentiment, readability, and linguistic metrics for pre-extracted articles.

Key Features:

-- Loads stopwords and positive/negative word dictionaries.

-- Tokenizes text into sentences and words, removing stopwords.

-- Calculates sentiment metrics:

-- Positive/Negative Score

-- Polarity Score

-- Subjectivity Score

-- Computes readability metrics:

-- Average Sentence Length

-- Percentage of Complex Words

-- FOG Index

-- Average Words per Sentence

-- Syllables per Word

-- Computes linguistic metrics:

-- Complex Word Count

-- Personal Pronouns Count

-- Average Word Length

Appends all metrics to the original input Excel and saves as Excel/CSV.
 
The output includes:
- Clean `.txt` article files.
- A structured Excel report (`output.xlsx`) containing all computed metrics.
- Logs of the extraction process.

How to Run:
run the following commands:
1. python scripts/extract_articles.py --input data/Input.xlsx --outdir articles --log output/extract_log.csv
2. python scripts/analyze_articles.py --input data/Input.xlsx --articles_dir articles --stopwords_dir data/StopWords --dict_dir data/MasterDictionary --output output/output.xlsx
 

Dependencies[requirements.txt]:
- pandas
- openpyxl
- requests
- beautifulsoup4
- nltk


