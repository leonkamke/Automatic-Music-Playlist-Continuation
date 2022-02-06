import os

import numpy as np
from os import listdir
from os import path
import argparse
import csv
import json

if __name__ == '__main__':
    data_path = './data/spotify_million_playlist_dataset/data'
    parser = argparse.ArgumentParser(description="get track_uri from playlist 198000")
    parser.add_argument(data_path, default=None)
    # print(listdir(data_path))
    file = path.join(data_path, 'mpd.slice.198000-198999.json')
    playlist_uris = []
    with open(file) as json_file:
        json_slice = json.load(json_file)
        playlist = json_slice['playlists']['pid' == 198000]
        for track in playlist['tracks']:
            playlist_uris.append(track['track_uri'])

    playlist_uris = np.array(playlist_uris)
    print(playlist_uris)
    print(type(playlist_uris))

