from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import pandas as pd
from taipy.gui import Gui, notify
import time

# Initialize the model and tokenizer
MODEL = "sbcBI/sentiment_analysis_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize variables
text = "Original text"
path = ""
treatment = 0

dataframe = pd.DataFrame(columns=["Text", "Score Pos", "Score Neu", "Score Neg", "Overall"])
dataframe2 = pd.DataFrame(columns=["Text", "Score Pos", "Score Neu", "Score Neg", "Overall"])

def analyze_text(input_text: str) -> dict:
    max_length = tokenizer.model_max_length
    encoded_text = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return {
        "Text": input_text[:50],
        "Score Pos": float(scores[2]),
        "Score Neu": float(scores[1]),
        "Score Neg": float(scores[0]),
        "Overall": float(scores[2] - scores[0]),
    }

def local_callback(state):
    notify(state, "Info", f"The text is: {state.text}", True)
    scores = analyze_text(state.text)
    state.dataframe = state.dataframe.append(scores, ignore_index=True)
    state.text = ""

def analyze_file(state):
    state.dataframe2 = pd.DataFrame(columns=["Text", "Score Pos", "Score Neu", "Score Neg", "Overall"])
    state.treatment = 0
    try:
        with open(state.path, "r", encoding="utf-8") as f:
            file_list = f.read().splitlines()

        total_lines = len(file_list)
        temp_df = pd.DataFrame(columns=["Text", "Score Pos", "Score Neu", "Score Neg", "Overall"])
        
        for i, input_text in enumerate(file_list):
            if input_text.strip():  # Skip empty lines
                scores = analyze_text(input_text)
                temp_df = temp_df.append(scores, ignore_index=True)
                
                # Update progress
                state.treatment = int((i + 1) * 100 / total_lines)
                
                # Update GUI every 50 lines or at the end
                if (i + 1) % 50 == 0 or i == len(file_list) - 1:
                    state.dataframe2 = state.dataframe2.append(temp_df, ignore_index=True)
                    temp_df = pd.DataFrame(columns=["Text", "Score Pos", "Score Neu", "Score Neg", "Overall"])
                    time.sleep(0.1)  # Small delay to allow GUI update
                    yield
        
        state.path = None
        notify(state, "Success", "File analysis completed!", True)
    except Exception as e:
        notify(state, "Error", f"Error processing file: {str(e)}", True)

# GUI definitions
page = """
# Sentiment Analysis with **Taipy**{: .color-primary} **GUI**{: .color-primary}

<|layout|columns=1 1|
<|
**My text:** <|{text}|>

**Enter a word:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>

<|Table|expandable|
<|{dataframe}|table|width=100%|number_format=%.2f|>
|>
|>

<|layout|columns=1 1 1|
## Positive <|{dataframe['Score Pos'].mean() if not dataframe.empty else 0}|text|format=%.2f|raw|>
## Neutral <|{dataframe['Score Neu'].mean() if not dataframe.empty else 0}|text|format=%.2f|raw|>
## Negative <|{dataframe['Score Neg'].mean() if not dataframe.empty else 0}|text|format=%.2f|raw|>
|>

<|{dataframe}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|>
"""

page_file = """
<|{path}|file_selector|extensions=.txt|label=Upload .txt file|on_action=analyze_file|> <|{f'Processing: {treatment}%'}|>

<br/>

<|Table|expandable|
<|{dataframe2}|table|width=100%|number_format=%.2f|>
|>

<br/>

<|{dataframe2}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|height=600px|>
"""

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "line": page,
    "text": page_file,
}

Gui(pages=pages).run(title="Sentiment Analysis")