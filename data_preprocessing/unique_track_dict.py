import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/spotify_million_playlist_dataset_csv/data/track_artist_dict.csv",
                     header=None, index_col=None)
    print(df.size)
    df = df.drop_duplicates()
    print(df.size)
    df.to_csv("../data/spotify_million_playlist_dataset_csv/data/track_artist_dict_unique.csv",
              header=False, index=False)

