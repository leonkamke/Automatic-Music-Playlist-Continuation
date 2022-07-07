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
    num_dimensions = 3
    num_vectors = 6000
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
    model = gensim.models.Word2Vec.load("../../models/gensim_word2vec/1_mil_playlists/word2vec-song-vectors.model")
    print("model loaded from file")

    a = model.wv.get_vector("spotify:track:7lQ8MOhq6IN2w8EYcFNSUk")
    b = model.wv.get_vector("spotify:track:6sDQ4uiWw9OdVrCXFLSlZt")
    c = model.wv.get_vector("spotify:track:1Slwb6dOYkBlWal1PGtnNg")
    d = model.wv.get_vector("spotify:track:1VdZ0vKfR5jneCmWIUAMxK")
    vectors = np.array([a, b, c, d])

    print(model.wv)

    print(vectors.shape)

    """# extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)[0:num_vectors]
    # vectors.shape = (num_vectors, 100)
    labels = np.asarray(model.wv.index_to_key)[0:num_vectors]  # fixed-width numpy strings"""

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)  # sehr rechenlastig
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]

    plt.figure(figsize=(15, 15))
    plt.scatter(x_vals, y_vals)
    plt.show()
