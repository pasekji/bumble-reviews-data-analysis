import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import deepl
from textblob import TextBlob
import spacy
import warnings
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bumble_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

class BumbleReviewAnalyzer:
    """
    Analyzer for Bumble app reviews implementing all steps from the conception phase.
    
    Implements a complete pipeline for text review analysis:
    1. Data loading and preprocessing
    2. Text cleaning and normalization
    3. Vectorization using BoW and TF-IDF
    4. Topic extraction using LDA and LSA
    
    Attributes:
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer for word normalization
        translator (deepl.Translator): DeepL translator for non-English texts
        nlp (spacy.Language): spaCy model for advanced NLP operations
    """

    def __init__(self, deepl_api_key):
        """
        Initialize the analyzer and load required models and tools.
        
        Args:
            deepl_api_key (str): Your DeepL API key
            
        Raises:
            Exception: If initialization of any component fails
        """
        logger.info("Initializing BumbleReviewAnalyzer...")
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.translator = deepl.Translator(deepl_api_key)
            self.nlp = spacy.load('en_core_web_sm')
            
            # Domain-specific stop words
            self.domain_stop_words = {
                'bumble', 'app', 'dating', 'would', 'one', 'time', 
                'really', 'want', 'even', 'also', 'make', 'way', 'since', 'go'
            }
            
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def load_data(self, file_path):
        """
        Load and preprocess the dataset according to conception phase.
        
        Steps:
        1. Load CSV file
        2. Remove unnecessary columns
        3. Remove duplicates
        4. Calculate basic statistics
        
        Args:
            file_path (str): Path to the CSV file containing reviews
            
        Returns:
            pd.DataFrame: Preprocessed dataset
            
        Raises:
            Exception: If data loading or preprocessing fails
        """
        logger.info(f"Loading data from {file_path}")
        try:
            # Load data
            df = pd.read_csv(file_path)
            initial_size = len(df)
            logger.info(f"Loaded {initial_size} initial reviews")

            # Remove unnecessary columns
            keep_cols = ['content', 'score']
            df = df[keep_cols]
            logger.info(f"Kept columns: {', '.join(keep_cols)}")

            # Remove duplicates
            df = df.drop_duplicates(subset=['content'])
            removed_duplicates = initial_size - len(df)
            logger.info(f"Removed {removed_duplicates} duplicate reviews ({removed_duplicates/initial_size*100:.2f}%)")

            # Basic statistics
            logger.info("\nInitial Dataset Statistics:")
            logger.info(f"Total reviews: {len(df)}")
            logger.info(f"Average rating: {df['score'].mean():.2f}")
            logger.info(f"Rating distribution:\n{df['score'].value_counts().sort_index()}")

            return df

        except Exception as e:
            logger.error(f"Error during data loading: {str(e)}")
            raise

    def translate_text(self, text):
        """
        Translate only non-English text using DeepL API.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Translated text if non-English, original text if English
            
        Note:
            Uses langdetect for language detection and only translates non-English text
        """
        try:
            # Check if text is empty or too short
            if not text or len(text.strip()) < 3:
                return text
                
            # Detect language
            try:
                detected_lang = detect(text)
                # Only translate if text is not in English
                if detected_lang != 'en':
                    logger.debug(f"Detected non-English text ('{detected_lang}'). Translating...")
                    result = self.translator.translate_text(text, target_lang="EN-US")
                    return result.text
                return text  # Return original text if it's English
                
            except LangDetectException as e:
                logger.warning(f"Language detection failed: {str(e)}")
                return text
                
        except Exception as e:
            logger.warning(f"Translation process failed: {str(e)}")
            return text

    def fix_spelling(self, text):
        """
        Fix spelling errors in the text using TextBlob.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with corrected spelling
            
        Note:
            If spelling correction fails, returns original text
        """
        try:
            text_blob = TextBlob(text)
            return str(text_blob.correct())
        except Exception as e:
            logger.warning(f"Spelling correction failed: {str(e)}")
            return text

    def clean_text(self, text):
        """
        Clean and normalize text according to conception phase steps.
        
        Steps:
        1. Remove HTML and URLs
        2. Translate non-English text
        3. Convert to lowercase
        4. Remove punctuation
        5. Fix spelling
        6. Remove stop words and domain-specific words
        7. Lemmatization
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML tags and URL
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Translation of non-English text
        text = self.translate_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Fix spelling
        text = self.fix_spelling(text)
        
        # Remove stop words and domain-specific words
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        stop_words.update(self.domain_stop_words)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatization (using both NLTK and spaCy)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        doc = self.nlp(' '.join(tokens))
        tokens = [token.lemma_ for token in doc]

        return ' '.join(tokens)

    def process_reviews(self, df):
        """
        Process all reviews in the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            
        Returns:
            pd.DataFrame: DataFrame with added cleaned_content column
        """
        logger.info("\nProcessing reviews...")
        try:
            # Batch processing
            batch_size = 1000
            processed_texts = []
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", total=total_batches):
                batch = df['content'].iloc[i:i+batch_size]
                processed_batch = [self.clean_text(text) for text in batch]
                processed_texts.extend(processed_batch)
                
                if i % (batch_size * 10) == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(df)} reviews")

            df['cleaned_content'] = processed_texts
            
            # Remove empty and short reviews
            initial_size = len(df)
            df = df[df['cleaned_content'].str.len() > 10]  # Remove very short reviews
            removed = initial_size - len(df)
            
            logger.info(f"Removed {removed} reviews (empty or too short)")
            logger.info(f"Final number of reviews: {len(df)}")
            
            # Review length statistics
            df['review_length'] = df['cleaned_content'].str.len()
            logger.info(f"Average review length: {df['review_length'].mean():.2f} characters")
            logger.info(f"Median review length: {df['review_length'].median()} characters")

            return df
            
        except Exception as e:
            logger.error(f"Error during review processing: {str(e)}")
            raise

    def vectorize_texts(self, texts):
        """
        Vectorize texts using both BoW and TF-IDF methods with optimized parameters.
        
        Args:
            texts (list): List of cleaned texts
            
        Returns:
            dict: Dictionary containing vectorized representations
        """
        logger.info("\nVectorizing texts...")
        try:
            # Bag of Words with optimized parameters
            logger.info("Applying BoW vectorization...")
            bow_vectorizer = CountVectorizer(
                max_features=1000,
                min_df=10,     # ignore rare terms
                max_df=0.9,    # ignore very common terms
                token_pattern=r'\b\w+\b'  # better token matching
            )
            bow_matrix = bow_vectorizer.fit_transform(texts)
            
            # TF-IDF with same parameters
            logger.info("Applying TF-IDF vectorization...")
            tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=10,
                max_df=0.9,
                token_pattern=r'\b\w+\b'
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            
            logger.info(f"Vocabulary size: {len(bow_vectorizer.get_feature_names_out())} terms")
            
            return {
                'bow': {'vectorizer': bow_vectorizer, 'matrix': bow_matrix},
                'tfidf': {'vectorizer': tfidf_vectorizer, 'matrix': tfidf_matrix}
            }
            
        except Exception as e:
            logger.error(f"Error during vectorization: {str(e)}")
            raise

    def extract_topics(self, vectors, texts):
        """
        Extract topics using both LDA and LSA methods with improved interpretation.
        
        Args:
            vectors (dict): Dictionary containing vectorized texts
            texts (list): List of cleaned texts
            
        Returns:
            dict: Dictionary containing topics and their importance scores
        """
        logger.info("\nExtracting topics...")
        try:
            # LDA with optimized parameters
            logger.info("Running LDA analysis...")
            lda = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                max_iter=20,
                learning_decay=0.7,
                learning_offset=50.
            )
            lda_output = lda.fit_transform(vectors['bow']['matrix'])
            
            # Extract LDA topics with importance scores
            feature_names = vectors['bow']['vectorizer'].get_feature_names_out()
            lda_topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[:-15:-1]  # Get top 15 words
                top_words = [(feature_names[i], topic[i]) for i in top_indices]
                lda_topics[f'Topic {topic_idx + 1}'] = top_words
            
            # LSA with optimized parameters
            logger.info("Running LSA analysis...")
            lsa = TruncatedSVD(
                n_components=5,
                random_state=42,
                algorithm='randomized'
            )
            lsa_output = lsa.fit_transform(vectors['tfidf']['matrix'])
            
            # Extract LSA topics with importance scores
            feature_names = vectors['tfidf']['vectorizer'].get_feature_names_out()
            lsa_topics = {}
            for topic_idx, topic in enumerate(lsa.components_):
                top_indices = topic.argsort()[:-15:-1]  # Get top 15 words
                top_words = [(feature_names[i], abs(topic[i])) for i in top_indices]
                lsa_topics[f'Topic {topic_idx + 1}'] = top_words
            
            logger.info("Topic extraction completed")
            
            return {
                'lda': {'topics': lda_topics, 'output': lda_output},
                'lsa': {'topics': lsa_topics, 'output': lsa_output}
            }
            
        except Exception as e:
            logger.error(f"Error during topic extraction: {str(e)}")
            raise

    def analyze_reviews(self, file_path):
        """
        Main analysis pipeline.
        
        Args:
            file_path (str): Path to the reviews file
            
        Returns:
            tuple: (processed_dataframe, vectorized_texts, extracted_topics)
        """
        start_time = datetime.now()
        logger.info(f"Starting Bumble reviews analysis at {start_time}")
        
        try:
            # Load and process data
            df = self.load_data(file_path)
            df = self.process_reviews(df)
            
            # Vectorize texts
            vectors = self.vectorize_texts(df['cleaned_content'])
            
            # Extract topics
            topics = self.extract_topics(vectors, df['cleaned_content'])
            
            # Log results with importance scores
            logger.info("\nExtracted Topics with Importance Scores:")
            
            logger.info("\nLDA Topics:")
            for topic, words in topics['lda']['topics'].items():
                logger.info(f"\n{topic}:")
                for word, score in words:
                    logger.info(f"  - {word} (importance: {score:.3f})")
            
            logger.info("\nLSA Topics:")
            for topic, words in topics['lsa']['topics'].items():
                logger.info("\nLSA Topics:")
                for topic, words in topics['lsa']['topics'].items():
                    logger.info(f"\n{topic}:")
                    for word, score in words:
                        logger.info(f"  - {word} (importance: {score:.3f})")
                
                end_time = datetime.now()
                duration = end_time - start_time
                logger.info(f"\nAnalysis completed in {duration}")
                
                return df, vectors, topics
                
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Get DeepL API key from environment variable for security
        deepl_api_key = os.getenv('DEEPL_API_KEY')
        if not deepl_api_key:
            raise ValueError("Please set DEEPL_API_KEY environment variable")
            
        analyzer = BumbleReviewAnalyzer(deepl_api_key)
        df, vectors, topics = analyzer.analyze_reviews("data/bumble_reviews.csv")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")