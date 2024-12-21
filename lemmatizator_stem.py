import pandas as pd

def lemmatize_all_eng(text_file, include_interviewer=True):
    """
    Basic example for English text lemmatization or cleaning.
    You can expand or adjust as needed.
    """
    with open(text_file) as f:
        text = f.readlines()

    if not include_interviewer:
        text = [line for line in text if not line.strip().startswith("Interviewer:")]

    lines = '\n'.join(text)
    paragraphs = lines.split('\n')
    df = pd.DataFrame(paragraphs, columns=['paragraphs'], index=range(1, len(paragraphs)+1))
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv('df_raw.csv')
    df['paragraphs'] = df['paragraphs'].str.lower()

    # Put your English lemmatization or any text cleaning logic here
    # For the demo, just saving as is
    df.to_excel('interview_lemmatized.xlsx', index=False)
    return df
