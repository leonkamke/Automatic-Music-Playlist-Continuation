import gensim
import spacy
import en_core_web_trf


if __name__ == "__main__":
    """print("load FastText model")
    fasttext_model = gensim.models.fasttext.load_facebook_vectors("models/fasttext/cc.en.300.bin")
    print("loaded model")
    word1 = "RAVE"
    word2 = "party playlist"
    word3 = "New School"
    print(fasttext_model.similarity("party", "party playlist"))
    print(fasttext_model.similar_by_vector(word1))
    print(fasttext_model.similar_by_vector(word2))
    print(fasttext_model.similar_by_vector(word3))
    """
    spacy.prefer_gpu()
    nlp = en_core_web_trf.load()
    doc1 = nlp("I like salty fries and hamburgers.")
    doc2 = nlp("Fast food tastes very good.")

    # Similarity of two documents
    print(doc1, "<->", doc2, doc1.similarity(doc2))
