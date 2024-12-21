def getText(text_file):
    with open(text_file) as f:
        text = f.readlines()
    lines = '\n'.join(text)
    return lines

def read_additional_stopwords(file_path):
    with open(file_path, "r") as file:
        # Splitting by comma since your stopwords are comma-separated
        additional_stopwords = file.read().split(',')
    return additional_stopwords

additional_stopwords = read_additional_stopwords('additional_stopwords.txt')

def preprocess_all(text_files, add_stop_words):
    import nltk
    import gensim
    import pandas as pd
    from gensim import corpora
    import pickle

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    # For a simple demo, let’s only do English
    from stop_words import get_stop_words
    en = get_stop_words('english')
    all_sw = en + additional_stopwords

    def preprocess(text_file, lemmatized_excel_file, length_restrict, bigram_mincount, all_stops):
        lines = getText(text_file).lower().split('\n')
        df = pd.DataFrame(lines, columns=['paragraphs'])
        df = df.drop_duplicates().reset_index(drop=True)
        df.to_csv('df_raw.csv')

        # Tokenize
        docs_tokenized = [row.split() for row in df['paragraphs']]
        
        # Remove stopwords
        x_eng_c = []
        stop_words_set = set(all_stops)
        for document in docs_tokenized:
            cleared_doc = [word for word in document if word not in stop_words_set and len(word) >= length_restrict]
            x_eng_c.append(cleared_doc)

        # Bigram
        bigram = gensim.models.Phrases(x_eng_c, min_count=bigram_mincount, threshold=40)
        x_eng_c = [bigram[line] for line in x_eng_c]

        dictionary = corpora.Dictionary(x_eng_c)
        corpus = [dictionary.doc2bow(text) for text in x_eng_c]

        # Optionally save some results for reference
        dictionary.save('dictionary')
        corpora.MmCorpus.serialize('corpus', corpus)

        # Save text for node referencing
        with open('x_train_alligned', 'wb') as f:
            pickle.dump([' '.join(doc) for doc in x_eng_c], f)

        df.to_excel(lemmatized_excel_file, index=False)
        return df, dictionary, corpus

    df_raw, dictionary, corpus = preprocess(
        text_files, 
        'interview_lemmatized.xlsx', 
        length_restrict=2, 
        bigram_mincount=3, 
        all_stops=all_sw
    )

    print("Preprocessing complete!")
    return df_raw
