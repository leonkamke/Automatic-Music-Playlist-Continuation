import pickle

import gensim


OUTPUT_PATH = "/netscratch/kamke/dictionaries/"


def build_track_uri2id():
    print("build track_uri2id dict")
    track2vec = gensim.models.Word2Vec.load('/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(track2vec.wv)):
        print(str(i))
        id = i
        uri = track2vec.wv.index_to_key[id]
        dict[uri] = id
    with open(OUTPUT_PATH + 'track_uri2id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


if __name__ == "__main__":
    build_track_uri2id()
