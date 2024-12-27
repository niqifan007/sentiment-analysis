from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
from taipy.gui import Gui, notify, State, download
from wordcloud import WordCloud
import io
import time
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import itertools
from langdetect import detect
import langdetect

# Download stopwords for multiple languages
nltk.download('stopwords', quiet=True)

# Define supported languages
SUPPORTED_LANGUAGES = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'russian', 'arabic', 'japanese']

# Create a dictionary mapping languages to their stopwords
LANGUAGE_STOPWORDS = {}

# Initialize NLTK supported languages' stopwords
for lang in SUPPORTED_LANGUAGES:
    if lang in stopwords.fileids():
        LANGUAGE_STOPWORDS[lang] = set(stopwords.words(lang))

# Language code mapping
LANG_CODE_MAP = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'ru': 'russian',
    'ar': 'arabic',
    'ja': 'japanese'
}

def get_stopwords(text):
    """Return the appropriate stopwords based on the language of the text"""
    try:
        # Detect the language of the text
        lang_code = detect(text)
        # Convert language code to the supported format
        lang = LANG_CODE_MAP.get(lang_code, 'english')  # Default to English
        return LANGUAGE_STOPWORDS.get(lang, LANGUAGE_STOPWORDS['english'])
    except langdetect.LangDetectException:
        # If language detection fails, return English stopwords
        return LANGUAGE_STOPWORDS['english']

# Initialize the model and tokenizer
MODEL = "sohan-ai/sentiment-analysis-model-amazon-reviews"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(MODEL)

# Initialize variables
text = "Original text"
path = ""
treatment = 0

# Define the columns for our DataFrames
columns = ["Text", "Full Text", "Score Positive", "Score Negative", "Overall"]
display_columns = ["Text", "Score Positive", "Score Negative", "Overall"]

dataframe = pd.DataFrame(columns=columns)
dataframe2 = pd.DataFrame(columns=columns)
sentiment_distribution = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})
sentiment_distribution2 = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})

# Initialize word cloud image variables
positive_wordcloud_img = None
negative_wordcloud_img = None

# Initialize LDA results
lda_results_positive = pd.DataFrame()
lda_results_negative = pd.DataFrame()

def analyze_text(input_text: str) -> dict:
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    scores = scores.detach().numpy()[0]

    return {
        "Text": input_text[:50],        # Truncated text for display
        "Full Text": input_text,        # Full text for export
        "Score Positive": float(scores[1]),
        "Score Negative": float(scores[0]),
        "Overall": "Positive" if scores[1] > scores[0] else "Negative",
    }

def local_callback(state: State):
    notify(state, "Info", f"The text is: {state.text}", True)
    scores = analyze_text(state.text)
    new_row = pd.DataFrame([scores], columns=columns)
    state.dataframe = pd.concat([state.dataframe.dropna(axis=1, how='all'), new_row.dropna(axis=1, how='all')], ignore_index=True)
    state.sentiment_distribution = get_sentiment_distribution(state.dataframe)
    state.text = ""
    # Update word clouds
    generate_wordclouds(state)

def analyze_file(state: State):
    if not state.path:
        notify(state, "Error", "No file selected", True)
        return

    state.dataframe2 = pd.DataFrame(columns=columns)
    state.treatment = 0

    try:
        with open(state.path, "r", encoding="utf-8") as f:
            file_list = f.readlines()

        total_lines = len(file_list)

        for i, input_text in enumerate(file_list):
            if input_text.strip():  # Skip empty lines
                scores = analyze_text(input_text.strip())
                new_row = pd.DataFrame([scores], columns=columns)
                state.dataframe2 = pd.concat([state.dataframe2.dropna(axis=1, how='all'), new_row.dropna(axis=1, how='all')], ignore_index=True)

                # Update progress
                state.treatment = int((i + 1) * 100 / total_lines)

                # Update GUI every 10 lines
                if (i + 1) % 10 == 0:
                    time.sleep(0.1)  # Small delay to allow GUI update

        state.treatment = 100  # Ensure we reach 100%
        state.sentiment_distribution2 = get_sentiment_distribution(state.dataframe2)
        notify(state, "Success", "File analysis completed!", True)
        # Update word clouds
        generate_wordclouds(state, from_file=True)
    except Exception as e:
        notify(state, "Error", f"Error processing file: {str(e)}", True)
    finally:
        state.path = None

def get_sentiment_distribution(df):
    if df.empty:
        return pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})
    return pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Count': [
            df[df['Overall'] == 'Positive'].shape[0],
            df[df['Overall'] == 'Negative'].shape[0]
        ]
    })

def download_single_csv(state: State):
    if not state.dataframe.empty:
        export_df = state.dataframe.drop(columns=["Text"])  # Remove truncated "Text" column
        export_df = export_df.rename(columns={"Full Text": "Text"})  # Rename "Full Text" to "Text"
        csv_content = export_df.to_csv(index=False, encoding='utf-8-sig')
        download(state, content=csv_content.encode('utf-8-sig'), name='single_text_analysis.csv')
    else:
        notify(state, "Error", "No single text analysis data to download", True)

def download_file_csv(state: State):
    if not state.dataframe2.empty:
        export_df = state.dataframe2.drop(columns=["Text"])  # Remove truncated "Text" column
        export_df = export_df.rename(columns={"Full Text": "Text"})  # Rename "Full Text" to "Text"
        csv_content = export_df.to_csv(index=False, encoding='utf-8-sig')
        download(state, content=csv_content.encode('utf-8-sig'), name='file_analysis.csv')
    else:
        notify(state, "Error", "No file analysis data to download", True)

def generate_wordclouds(state: State, from_file=False):
    if from_file:
        df = state.dataframe2
    else:
        df = state.dataframe

    # Generate positive word cloud
    positive_texts = df[df['Overall'] == 'Positive']['Full Text']
    positive_text = ' '.join(positive_texts)
    if positive_text:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        state.positive_wordcloud_img = wordcloud_to_buffer(wordcloud_pos)
    else:
        state.positive_wordcloud_img = None

    # Generate negative word cloud
    negative_texts = df[df['Overall'] == 'Negative']['Full Text']
    negative_text = ' '.join(negative_texts)
    if negative_text:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        state.negative_wordcloud_img = wordcloud_to_buffer(wordcloud_neg)
    else:
        state.negative_wordcloud_img = None

def wordcloud_to_buffer(wordcloud):
    img_buffer = io.BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer.read()

def clear_results(state: State):
    # Reset the dataframes
    state.dataframe = pd.DataFrame(columns=columns)
    state.dataframe2 = pd.DataFrame(columns=columns)
    
    # Reset sentiment distributions
    state.sentiment_distribution = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})
    state.sentiment_distribution2 = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})
    
    # Reset word cloud images
    state.positive_wordcloud_img = None
    state.negative_wordcloud_img = None
    
    # Reset LDA results
    state.lda_results_positive = pd.DataFrame()
    state.lda_results_negative = pd.DataFrame()
    
    # Optionally reset other variables
    state.text = ""
    state.path = ""
    state.treatment = 0
    
    # Notify the user
    notify(state, "Info", "Results have been cleared.", True)

def on_init(state: State):
    state.dataframe = pd.DataFrame(columns=columns)
    state.dataframe2 = pd.DataFrame(columns=columns)
    state.sentiment_distribution = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})
    state.sentiment_distribution2 = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [0, 0]})
    state.positive_wordcloud_img = None
    state.negative_wordcloud_img = None
    state.lda_results_positive = pd.DataFrame()
    state.lda_results_negative = pd.DataFrame()
    state.text = ""
    state.path = ""
    state.treatment = 0

# LDA Analysis Functions
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    # Use the stopwords of the detected language
    stop_words = get_stopwords(text)
    return [token for token in tokens if token not in stop_words]

def perform_lda(texts, n_topics=15):
    processed_data = [" ".join(preprocess(text)) for text in texts]
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(2, 2))
    data_vectorized = vectorizer.fit_transform(processed_data)
    
    lda = LDA(n_components=n_topics, random_state=0)
    lda.fit(data_vectorized)
    
    tf_feature_names = vectorizer.get_feature_names_out()
    topics = display_topics(lda, tf_feature_names, 15)
    
    topic_counts = lda.transform(data_vectorized).argmax(axis=1)
    topic_count_df = pd.DataFrame(topic_counts, columns=["Topic"])
    topic_count_df["Count"] = 1
    topic_count_summary = topic_count_df.groupby("Topic").count().reset_index()
    
    topic_summary = pd.DataFrame(topics, columns=["Topic", "Representation"])
    topic_summary["Topic"] = topic_summary["Topic"].str.extract(r'(\d+)').astype(int)
    
    result_df = pd.merge(topic_count_summary, topic_summary, on="Topic", how="left")
    result_df.columns = ["Topic", "Count", "Representation"]
    result_df["Name"] = result_df.apply(lambda row: " ".join(row['Representation'].split(',')[:2]), axis=1)
    
    return result_df[["Topic", "Count", "Name", "Representation"]]

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([f"Topic {topic_idx}",
                        ", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])])
    return topics

def lda_analyze_positive(state: State):
    if state.dataframe2.empty:
        notify(state, "Error", "No data available for LDA analysis", True)
        return
    positive_texts = state.dataframe2[state.dataframe2['Overall'] == 'Positive']['Full Text']
    state.lda_results_positive = perform_lda(positive_texts)
    notify(state, "Success", "LDA analysis completed for positive reviews!", True)

def lda_analyze_negative(state: State):
    if state.dataframe2.empty:
        notify(state, "Error", "No data available for LDA analysis", True)
        return
    negative_texts = state.dataframe2[state.dataframe2['Overall'] == 'Negative']['Full Text']
    state.lda_results_negative = perform_lda(negative_texts)
    notify(state, "Success", "LDA analysis completed for negative reviews!", True)

def download_lda_results(state: State, sentiment):
    if sentiment == 'positive' and not state.lda_results_positive.empty:
        csv_content = state.lda_results_positive.to_csv(index=False, encoding='utf-8-sig')
        download(state, content=csv_content.encode('utf-8-sig'), name='lda_results_positive.csv')
    elif sentiment == 'negative' and not state.lda_results_negative.empty:
        csv_content = state.lda_results_negative.to_csv(index=False, encoding='utf-8-sig')
        download(state, content=csv_content.encode('utf-8-sig'), name='lda_results_negative.csv')
    else:
        notify(state, "Error", f"No LDA results available for {sentiment} reviews", True)

def download_lda_positive(state: State):
    download_lda_results(state, 'positive')

def download_lda_negative(state: State):
    download_lda_results(state, 'negative')

# GUI definitions
page = """
# Amazon Reviews **Sentiment**{: .color-primary} **Analysis**{: .color-primary}

<|layout|columns=1 1|
<|
**My Text:** <|{text}|>

**Enter a text:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>

<|Table|expandable|
<|{dataframe}|table|columns={display_columns}|width=100%|number_format=%.2f|>
|>
<|{None}|file_download|label=Download CSV File|on_action=download_single_csv|>
<|Clear Results|button|on_action=clear_results|>

|>

<|layout|columns=1 1|
## Average Positive Score <|{dataframe['Score Positive'].mean() if not dataframe.empty else 0}|text|format=%.2f|raw|>
## Average Negative Score <|{dataframe['Score Negative'].mean() if not dataframe.empty else 0}|text|format=%.2f|raw|>
|>

<|{dataframe}|chart|type=bar|x=Text|y[1]=Score Positive|y[2]=Score Negative|color[1]=green|color[2]=red|>

<|layout|columns=1 1|
<|
### Positive Word Cloud
<|{positive_wordcloud_img}|image|width=100%|height=400px|>
|>
<|
### Negative Word Cloud
<|{negative_wordcloud_img}|image|width=100%|height=400px|>
|>
|>

<|tabs|
<|part|label=Positive|
### Positive Sentiments
<|{dataframe[dataframe['Overall'] == 'Positive']}|table|columns={display_columns}|width=100%|number_format=%.2f|>
|>
<|part|label=Negative|
### Negative Sentiments
<|{dataframe[dataframe['Overall'] =='Negative']}|table|columns={display_columns}|width=100%|number_format=%.2f|>
|>
|>

<|
### Sentiment Distribution
<|{sentiment_distribution}|chart|type=pie|values=Count|labels=Sentiment|>
|>
"""

page_file = """
<|{path}|file_selector|extensions=.txt|label=Upload .txt file|on_action=analyze_file|>

<|{f'Processing: {treatment}%'}|text|>
<|{treatment}|progress|>

<br/>

<|Table|expandable|
<|{dataframe2}|table|columns={display_columns}|width=100%|number_format=%.2f|>
|>
<|{None}|file_download|label=Download CSV File|on_action=download_file_csv|>
<|Clear Results|button|on_action=clear_results|>

<br/>

<|{dataframe2}|chart|type=bar|x=Text|y[1]=Score Positive|y[2]=Score Negative|color[1]=green|color[2]=red|height=600px|>

<|layout|columns=1 1|
<|
### Positive Word Cloud
<|{positive_wordcloud_img}|image|width=100%|height=400px|>
|>
<|
### Negative Word Cloud
<|{negative_wordcloud_img}|image|width=100%|height=400px|>
|>
|>

<|part|label=Positive|
### Positive Sentiments
<|{dataframe2[dataframe2['Overall'] == 'Positive']}|table|columns={display_columns}|width=100%|number_format=%.2f|>
|>

<|part|label=Negative|
### Negative Sentiments
<|{dataframe2[dataframe2['Overall'] == 'Negative']}|table|columns={display_columns}|width=100%|number_format=%.2f|>
|>

<|
### Sentiment Distribution
<|{sentiment_distribution2}|chart|type=pie|values=Count|labels=Sentiment|>
|>

<br/>

## LDA Analysis

<|layout|columns=1 1|
<|Analyze Positive Reviews|button|on_action=lda_analyze_positive|>
<|Analyze Negative Reviews|button|on_action=lda_analyze_negative|>
|>

<|layout|columns=1 1|
<|
### LDA Results for Positive Reviews
<|{lda_results_positive}|table|width=100%|>
<|Download LDA Results (Positive)|button|on_action=download_lda_positive|>
|>

<|
### LDA Results for Negative Reviews
<|{lda_results_negative}|table|width=100%|>
<|Download LDA Results (Negative)|button|on_action=download_lda_negative|>
|>
|>
"""

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "line": page,
    "text": page_file,
}

if __name__ == "__main__":
    Gui(pages=pages).run(title="Sentiment Analysis", on_init=on_init)
