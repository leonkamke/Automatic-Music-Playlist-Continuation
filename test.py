import numpy as np
import gensim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    print("load model from file")
    model = gensim.models.Word2Vec.load("./models/word2vec-song-vectors.model")
    print("model loaded from file")
    vectors = np.asarray(model.wv.vectors)

    # x = StandardScaler().fit_transform(vectors)

    pca = PCA(n_components=3)
    vectors_reduced1 = pca.fit_transform(vectors)
    plt.scatter(vectors_reduced1[:, 0], vectors_reduced1[:, 1], vectors_reduced1[:, 2])
    plt.show()


