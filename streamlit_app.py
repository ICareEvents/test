import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import torch
from transformers import pipeline

#############################################
# Load spaCy English model
#############################################
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

#############################################
# Summarization pipeline (HuggingFace)
#############################################
summarizer = pipeline(
    "summarization", 
    model="facebook/bart-base",
    tokenizer="facebook/bart-base"
)

def read_sample_text():
    """
    Reads from 'sample.txt' in the same directory.
    """
    try:
        with open("sample.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "sample.txt not found. Please place a file named sample.txt in the same directory."

def preprocess_text(text: str):
    """
    Splits text into paragraphs and preprocesses each.
    Returns:
        paragraphs (list[str]): original paragraphs
        processed_paras (list[str]): cleaned text paragraphs
        token_lists (list[list[str]]): tokenized paragraphs
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    lemmatizer = WordNetLemmatizer()
    eng_stop = set(stopwords.words('english'))

    processed_paras = []
    token_lists = []

    for para in paragraphs:
        tokens = word_tokenize(para.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in eng_stop]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        processed_paras.append(" ".join(tokens))
        token_lists.append(tokens)

    return paragraphs, processed_paras, token_lists

def extract_entities(text: str):
    """
    Extract named entities from text using spaCy. Returns list of (entity_text, entity_label).
    """
    if not nlp:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_sentiment(text: str):
    """
    TextBlob sentiment analysis. Returns (polarity, subjectivity).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def build_lda_topics(token_lists, num_topics=3, num_words=5):
    """
    Build LDA model using gensim. 
    Returns a dictionary of topic -> top words, and a pyLDAvis visualization object.
    """
    dictionary = corpora.Dictionary(token_lists)
    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]

    lda_model = models.LdaModel(
        corpus, 
        num_topics=num_topics, 
        id2word=dictionary, 
        passes=5, 
        random_state=42
    )

    topics = {}
    for i in range(num_topics):
        words_probs = lda_model.show_topic(i, topn=num_words)
        top_words = [wp[0] for wp in words_probs]
        topics[i] = top_words

    # Prepare pyLDAvis
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
    return topics, lda_display

def generate_summary(full_text: str, max_length=130):
    """
    Summarize the entire text using a huggingface model.
    """
    try:
        summary_result = summarizer(full_text, max_length=max_length, min_length=30, do_sample=False)
        return summary_result[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

###########################################
# STREAMLIT APP
###########################################
def main():
    st.title("Interview Analysis & Topic Modeling Demo")

    st.write("""
    - Upload your own `.txt` file **OR** leave it blank to use **sample.txt**.  
    - This prototype will:
      1. Extract named entities (e.g., people, orgs)  
      2. Perform sentiment analysis (positive/negative)  
      3. Do topic modeling with LDA (to reveal common themes)  
      4. Summarize the entire text in a short paragraph  
      5. Display an interactive **pyLDAvis** graph for the topics  
    """)

    # Upload file
    uploaded_file = st.file_uploader("Upload a text file here", type=["txt"])
    
    # Number of LDA topics
    num_topics = st.slider("Number of LDA topics", 2, 10, 3)
    # Number of words per topic
    num_words_topic = st.slider("Top words per topic", 3, 10, 5)

    if st.button("Run Analysis"):
        # Read from uploaded file or sample.txt
        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success("Using your uploaded file.")
        else:
            raw_text = read_sample_text()
            st.success("No file uploaded. Using sample.txt instead.")

        # Preprocess
        paragraphs, processed_paras, token_lists = preprocess_text(raw_text)

        st.subheader("Paragraph-by-Paragraph Analysis")
        for i, para in enumerate(paragraphs):
            st.markdown(f"**Paragraph {i+1}**:")
            st.write(para)
            # Entities
            ents = extract_entities(para)
            if ents:
                entity_str = ", ".join([f"{t}({lbl})" for t, lbl in ents])
            else:
                entity_str = "No named entities found"
            # Sentiment
            pol, subj = analyze_sentiment(para)
            st.write(f"**Named Entities**: {entity_str}")
            st.write(f"**Sentiment**: polarity={pol:.2f}, subjectivity={subj:.2f}")
            st.markdown("---")

        # LDA
        st.subheader("Topic Modeling (LDA)")
        with st.spinner("Building LDA model..."):
            topics, lda_display = build_lda_topics(token_lists, num_topics=num_topics, num_words=num_words_topic)
            st.success("LDA model built!")

        st.write("**Top words in each topic:**")
        for topic_id, words in topics.items():
            st.write(f"Topic {topic_id}: {', '.join(words)}")

        st.subheader("pyLDAvis Visualization")
        st.write("(An interactive LDA visualization. Please wait a moment for it to render.)")
        vis_html = pyLDAvis.prepared_data_to_html(lda_display)
        st.components.v1.html(vis_html, width=1300, height=800, scrolling=True)

        # Summarize
        st.subheader("Overall Summary")
        with st.spinner("Generating summary..."):
            summary_text = generate_summary(raw_text, max_length=130)
        st.write(summary_text)

        st.info("Analysis Complete!")

if __name__ == "__main__":
    main()
