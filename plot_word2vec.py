import numpy as np
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
loads the word2vec model and plots some dimensionality reduced song
vectors in a 2D coordinate system
"""

if __name__ == '__main__':
    # parameters vor vector plotting
    num_dimensions = 2
    num_vectors = 10000

    print("load model from file")
    model = gensim.models.Word2Vec.load("./models/word2vec-song-vectors.model")
    print("model loaded from file")

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)[0:num_vectors]
    labels = np.asarray(model.wv.index_to_key)[0:num_vectors]  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)  # sehr rechenlastig
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]

    plt.figure(figsize=(15, 15))
    plt.scatter(x_vals, y_vals)
    """
    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))    
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()
    """
    plt.show()