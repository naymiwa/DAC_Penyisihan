import re
import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

def extract_features(
    train,
    test,
    max_features=10000,
    ngram_range=(1, 2),
    vector_size=100,
    window_size=5,
    seed=42
):
    # ==========================================
    # 1. PERSIAPAN TF-IDF
    # ==========================================
    print("Membuat fitur TF-IDF...")
    all_text = (
        train["clean_title"].tolist()
        + train["clean_content"].tolist()
        + test["clean_title"].tolist()
        + test["clean_content"].tolist()
    )

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )
    tfidf.fit(all_text)

    train_title_tfidf = tfidf.transform(train["clean_title"])
    train_content_tfidf = tfidf.transform(train["clean_content"])
    test_title_tfidf = tfidf.transform(test["clean_title"])
    test_content_tfidf = tfidf.transform(test["clean_content"])

    # Simpan bobot IDF untuk mekanisme Attention/Weighted Word2Vec nanti
    idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    # ==========================================
    # 2. FITUR KEMIRIPAN DASAR
    # ==========================================
    print("Menghitung fitur Jaccard & Cosine...")

    def compute_jaccard(title, content):
        set_t = set(str(title).split())
        set_c = set(str(content).split())
        if not set_t or not set_c:
            return 0.0
        return len(set_t.intersection(set_c)) / len(set_t.union(set_c))

    for df, title_mat, content_mat in [
        (train, train_title_tfidf, train_content_tfidf),
        (test, test_title_tfidf, test_content_tfidf)
    ]:
        df["jaccard_sim"] = df.apply(
            lambda x: compute_jaccard(x["clean_title"], x["clean_content"]), axis=1
        )

        def row_wise_cosine_similarity(mat_a, mat_b):
            dot = mat_a.multiply(mat_b).sum(axis=1)
            norm_a = np.sqrt(mat_a.multiply(mat_a).sum(axis=1))
            norm_b = np.sqrt(mat_b.multiply(mat_b).sum(axis=1))
            denom = np.asarray(norm_a).flatten() * np.asarray(norm_b).flatten()
            denom[denom == 0] = 1e-10
            return np.asarray(dot).flatten() / denom

        df["tfidf_cosine_sim"] = row_wise_cosine_similarity(title_mat, content_mat)

        title_length = df["clean_title"].apply(lambda x: len(str(x).split()))
        content_length = df["clean_content"].apply(lambda x: len(str(x).split()))
        df["length_ratio"] = content_length / (title_length + 1.0)

    # ==========================================
    # 3. TF-IDF WEIGHTED WORD2VEC (PENGGANTI ATTENTION)
    # ==========================================
    print("Melatih model Word2Vec & menerapkan Weighted Pooling...")
    sentences = [str(text).split() for text in all_text]

    w2v_model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window_size,
        min_count=2,
        workers=4,
        seed=seed
    )

    def get_weighted_w2v_vector(text):
        words = str(text).split()
        vectors = []
        weights = []
        
        for word in words:
            if word in w2v_model.wv:
                vectors.append(w2v_model.wv[word])
                # Jika kata tidak ada di idf_dict, beri bobot standar 1.0
                weights.append(idf_dict.get(word, 1.0))
                
        if not vectors:
            return np.zeros(vector_size)
            
        # Menggunakan np.average untuk memberikan bobot berbeda (Attention Simulation)
        return np.average(vectors, axis=0, weights=weights)

    def compute_cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    for df in [train, test]:
        title_w2v = np.array([get_weighted_w2v_vector(text) for text in df["clean_title"]])
        content_w2v = np.array([get_weighted_w2v_vector(text) for text in df["clean_content"]])
        df["w2v_cosine_sim"] = [
            compute_cosine_similarity(t, c) for t, c in zip(title_w2v, content_w2v)
        ]

    # ==========================================
    # 4. FITUR HARD NEGATIVE (ANGKA, ENTITAS, NEGASI)
    # ==========================================
    print("Mengekstrak fitur Angka, Entitas, dan Negasi dari teks asli...")

    NUMBER_WORDS = {
        "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
        "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10"
    }

    NEGATION_WORDS = {
        "tidak", "bukan", "batal", "gagal", "tolak", "menolak",
        "bantah", "membantah", "sangkal", "menyangkal", "tepis",
        "menepis", "hoaks", "salah", "klarifikasi", "keliru", "palsu"
    }

    def normalize_number_words(text):
        return " ".join(NUMBER_WORDS.get(w, w) for w in str(text).lower().split())

    def extract_numbers(text):
        return set(re.findall(r'\d+', normalize_number_words(text)))

    def compute_number_overlap(title, content):
        nums_t = extract_numbers(title)
        nums_c = extract_numbers(content)
        if not nums_t: return 1.0
        return len(nums_t & nums_c) / len(nums_t)

    def extract_capitalized_words(text):
        words = str(text).split()
        caps = set()
        for i, w in enumerate(words):
            clean_w = re.sub(r'[^A-Za-z]', '', w)
            if len(clean_w) > 1 and clean_w[0].isupper() and i != 0:
                caps.add(clean_w.lower())
        return caps

    def compute_entity_overlap(title, content):
        caps_t = extract_capitalized_words(title)
        caps_c = extract_capitalized_words(content)
        if not caps_t: return 1.0
        return len(caps_t & caps_c) / len(caps_t)

    def has_negation(text):
        return 1 if set(str(text).lower().split()) & NEGATION_WORDS else 0

    for df in [train, test]:
        # Fitur Overlap Angka & Entitas
        df["number_overlap"] = df.apply(lambda x: compute_number_overlap(x["title"], x["content"]), axis=1)
        df["entity_overlap"] = df.apply(lambda x: compute_entity_overlap(x["title"], x["content"]), axis=1)
        
        # Fitur Negation Mismatch
        df["title_negation"] = df["title"].apply(has_negation)
        df["content_negation"] = df["content"].apply(has_negation)
        df["negation_mismatch"] = (df["title_negation"] != df["content_negation"]).astype(int)

    print("Feature Engineering Selesai!")

    # ==========================================
    # 5. PENGGABUNGAN MATRIKS
    # ==========================================
    extra_cols = [
        "jaccard_sim",
        "tfidf_cosine_sim",
        "length_ratio",
        "w2v_cosine_sim",
        "number_overlap",
        "entity_overlap",
        "negation_mismatch"
    ]

    X_train = sp.hstack([
        train_title_tfidf,
        train_content_tfidf,
        train[extra_cols].values
    ]).tocsr()

    X_test = sp.hstack([
        test_title_tfidf,
        test_content_tfidf,
        test[extra_cols].values
    ]).tocsr()

    return train, test, X_train, X_test, tfidf, w2v_model