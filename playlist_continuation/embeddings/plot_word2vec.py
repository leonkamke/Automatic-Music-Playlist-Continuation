import numpy as np
import gensim
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
loads the word2vec model and plots some dimensionality reduced song
vectors in a 2D coordinate system
"""


def get_index(vector, matrix):
    for i, row in enumerate(matrix):
        if vector[0] == row[0] and vector[1] == row[1] and vector[2] == row[2]:
            print("i = ", i)
            return i


if __name__ == '__main__':
    # parameters vor vector plotting
    num_dimensions = 3
    num_vectors = 700
    num_clusters = 10
    """
    
    Without me: "spotify:track:7lQ8MOhq6IN2w8EYcFNSUk"
    "artist_name": "Eminem", 
                    "track_uri": "spotify:track:6sDQ4uiWw9OdVrCXFLSlZt", 
                    "artist_uri": "spotify:artist:7dGJo4pcD2V6oG8kP0tJRR", 
                    "track_name": "Rap God", 
    
    "artist_name": "Ed Sheeran", 
                    "track_uri": "spotify:track:1Slwb6dOYkBlWal1PGtnNg", 
                    "artist_uri": "spotify:artist:6eUKZXaKkcviH0Ku9w2n3V", 
                    "track_name": "Thinking Out Loud", 
                    
  
                    "artist_name": "Ed Sheeran", 
                    "track_uri": "spotify:track:1VdZ0vKfR5jneCmWIUAMxK", 
                    "artist_uri": "spotify:artist:6eUKZXaKkcviH0Ku9w2n3V", 
                    "track_name": "The A Team"
    """
    print("load model from file")
    model = gensim.models.Word2Vec.load(
        "../../models/gensim_word2vec/1_mil_playlists/word2vec-song-vectors.model")
    print("model loaded from file")

    keys = model.wv.index_to_key[:num_vectors]
    vectors = []
    for key in keys:
        vectors.append(model.wv.get_vector(key))
    vectors = np.array(vectors)
    print(vectors.shape)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    # vectors.shape = (num_vectors, 3)
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]
    z_vals = vectors[:, 2]

    # apply k-means clustering
    k_means = KMeans(n_clusters=num_clusters)
    label = k_means.fit_predict(vectors)

    # plot the clusters
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    u_labels = np.unique(label)
    print("u_labels = ", u_labels)
    for i in u_labels:
        ax.scatter3D(vectors[label == i, 0], vectors[label == i, 1], vectors[label == i, 2], label=i)
    plt.legend()
    plt.show()

    id = get_index(vectors[label == 3][0], vectors)
    id1 = get_index(vectors[label == 3][1], vectors)
    id2 = get_index(vectors[label == 5][0], vectors)
    id3 = get_index(vectors[label == 5][1], vectors)

    print(model.wv.index_to_key[id])
    print(model.wv.index_to_key[id1])
    print(model.wv.index_to_key[id2])
    print(model.wv.index_to_key[id3])


