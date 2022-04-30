import pandas as pd

# skript for creating the vocabulary (list of all tracks)
if __name__ == '__main__':
    # read from items.csv wich already contains all tracks, but there are also listet duplicates
    df = pd.read_csv('data/spotify_million_playlist_dataset_csv/data/items.csv', header=None, usecols=[2])
    # delete duplicates and reset index
    df = df.drop_duplicates()
    df = df.reset_index()
    del df['index']
    df.info()
    # write into new csv file, with vocabulary
    df.to_csv('data/spotify_million_playlist_dataset_csv/data/vocabulary.csv', header=False)
