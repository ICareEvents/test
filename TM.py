import gensim
from sklearn.cluster import KMeans
import hdbscan
import umap

class Topic_Model:
    def __init__(self, k=5, method='LDA', num_keywords=10):
        self.k = k
        self.method = method
        self.num_keywords = num_keywords
        self.vec = {}  # for storing vector representations if using BERT

    def fit(self, corpus, dictionary, cluster_model=None):
        """
        Main fitting function for topic modeling.
        Supports methods 'LDA' or 'BERT',
        and clustering methods like 'KMeans' or 'HDBSCAN'.
        """
        if self.method == 'LDA':
            print('Fitting LDA ...')
            self.ldamodel = gensim.models.LdaMulticore(
                corpus, num_topics=self.k, id2word=dictionary, workers=2, passes=10, random_state=100
            )
            print('Fitting LDA Done!')
            # For demonstration, return some placeholder values
            return self.ldamodel, None

        elif self.method == 'BERT':
            print('Getting vector representations for BERT...')
            # Instead of language detection, assume English by default
            # Load an English BERT model (placeholder):
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")

            # Example data for demonstration – in real code, load from preprocessed text
            # This is just a placeholder approach to show how you’d do BERT embeddings
            # x_train is loaded from 'x_train_alligned'
            import pickle
            with open('x_train_alligned','rb') as f:
                x_train = pickle.load(f)  # list of strings

            def get_bert_vectors(texts, model, tokenizer):
                vectors = []
                for txt in texts:
                    inputs = tokenizer(txt, return_tensors='pt', truncation=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # We take the [CLS] token embedding as a simple approach
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    vectors.append(cls_embedding)
                return vectors

            vecs_bert = get_bert_vectors(x_train, model, tokenizer)
            self.vec['BERT'] = vecs_bert
            print('Getting vector representations for BERT. Done!')

            if cluster_model == 'hdbscan':
                print("Clustering with HDBSCAN ...")
                umap_embeddings = umap.UMAP(
                    n_neighbors=15, n_components=5, metric='cosine'
                ).fit_transform(vecs_bert)
                cm = hdbscan.HDBSCAN(
                    gen_min_span_tree=True, 
                    min_cluster_size=5, 
                    min_samples=6, 
                    metric='euclidean'
                ).fit(umap_embeddings)
                print("Clustering with HDBSCAN done!")
                # Return or store results as needed
                return cm
            else:
                # Return the embeddings if no cluster model specified
                return vecs_bert

        # Optionally handle KMeans, etc.
        # Just returning placeholders
        return None
