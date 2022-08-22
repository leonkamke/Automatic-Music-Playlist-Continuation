"""
loads the word2vec model and plots some dimensionality reduced song
vectors in a 2D coordinate system
"""

import numpy as np
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # parameters vor vector plotting
    num_dimensions = 3
    num_vectors = 6000
    print("load model from file")
    model = gensim.models.Word2Vec.load("../../models/gensim_word2vec/1_mil_playlists/word2vec-song-vectors.model")
    print("model loaded from file")

    a = model.wv.get_vector("spotify:track:7lQ8MOhq6IN2w8EYcFNSUk")
    b = model.wv.get_vector("spotify:track:6sDQ4uiWw9OdVrCXFLSlZt")
    c = model.wv.get_vector("spotify:track:1Slwb6dOYkBlWal1PGtnNg")
    d = model.wv.get_vector("spotify:track:1VdZ0vKfR5jneCmWIUAMxK")
    vectors = np.array([a, b, c, d])

    print(model.wv)
    print(vectors.shape)

    # extract the words & their vectors, as numpy arrays
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)  # computationally heavy
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]

    plt.figure(figsize=(15, 15))
    plt.scatter(x_vals, y_vals)
    plt.show()
