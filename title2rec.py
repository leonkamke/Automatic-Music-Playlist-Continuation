import gensim


if __name__ == "__main__":
    print("load FastText model")
    fasttext_model = gensim.models.fasttext.load_facebook_vectors("models/fasttext/cc.en.300.bin")
    print("loaded model")
    word1 = "RAVE"
    word2 = "party playlist"
    word3 = "New School"
    print(fasttext_model.similarity("party", "partyplaylist"))
    print(fasttext_model.similar_by_vector(word1))
    print(fasttext_model.similar_by_vector(word2))
    print(fasttext_model.similar_by_vector(word3))
