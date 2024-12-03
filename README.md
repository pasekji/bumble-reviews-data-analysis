# Bumble Reviews Analysis

NLP analysis of app reviews for Data Analysis course.

## Project Setup

1. Clone repository and create virtual environment:
```bash
git clone [repository-url]
cd bumble-analysis
python -m venv venv
source venv/bin/activate 
```

2. Install required packages:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

3. Set up DeepL API key:
```bash
export DEEPL_API_KEY=your-api-key
```

## Run Analysis

```bash
python src/analyzer.py
```

## Project Structure
```
bumble-analysis/
├── data/               # dataset directory
├── src/               # source code
│   └── analyzer.py    # main analyzer
├── outputs/           # analysis outputs
│   ├── bumble_analysis.log    # complete processing log
│   ├── results.txt            # results log
│   └── ratings_distribution.png  # ratings visualization
├── requirements.txt   # dependencies
└── README.md
``` 

## Author
Jiří Pašek  
DLBDSEDA02 Project: Data Analysis