def make_graph_big(method, k, num_keywords=20):
    from TM import Topic_Model
    from gensim import corpora
    import ast
    import json
    import warnings
    import random

    warnings.filterwarnings('ignore')

    dictionary = corpora.Dictionary.load('dictionary')
    corpus = corpora.MmCorpus('corpus')

    number_of_colors = 1000
    color_pallette = [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for i in range(number_of_colors)
    ]

    clear_nodes, clear_links = [], []

    if method == 'BERT':
        tm = Topic_Model(k=int(k), method='BERT', num_keywords=num_keywords)
        model_res = tm.fit(corpus, dictionary, cluster_model='hdbscan')
        # The logic to generate nodes/links is omitted for brevity
        # Suppose we fill in clear_nodes, clear_links from model_res

    elif method == 'LDA':
        tm1 = Topic_Model(k=int(k), method='LDA', num_keywords=num_keywords)
        model_res, to_show = tm1.fit(corpus, dictionary)
        # The logic to generate nodes/links is omitted for brevity
        # Suppose we fill in clear_nodes, clear_links from model_res / to_show

    print('Everything is ready. Please run the server or check the Vercel deployment.')
    return clear_nodes, clear_links

def translate_to_eng(clear_nodes, clear_links):
    from googletrans import Translator
    import json
    translator = Translator()

    eng_nodes = []
    for tupl in clear_nodes:
        for key, items in tupl.items():
            smol_dict = {'name': translator.translate(items).text}
            eng_nodes.append(smol_dict)

    dict_json = {'links': clear_links, 'nodes': eng_nodes}
    with open('graph.json', 'w', encoding='utf-8') as file:
        json.dump(dict_json, file, ensure_ascii=False)

    print('Translation done. Check graph.json.')
