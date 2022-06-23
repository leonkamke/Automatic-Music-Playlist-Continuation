import torch
import csv
from playlist_continuation.config import load_attributes as la

# training playlists:    0 - 980000
# evaluation playlists:  980001 - 999999
# --> evaluation_set contains of 20000 playlists
start_idx = 980001
end_idx = 999999
# here are 13934 playlists, with length >= 30 tracks

def create_dataset():
    with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create lists of track uri's

        """c_title_only = 0
        c_first_5 = 0
        c_first_10 = 0
        c_first_25 = 0
        c_first_100 = 0"""
        num_140_ = []
        num_100_140 = []
        num_60_100 = []
        num_30_60 = []

        num_playlists_more_30 = 0
        for index, row in enumerate(csv_reader):
            if start_idx <= index <= end_idx and len(row) >= 32:
                num_playlists_more_30 += 1
                len_playlist = len(row) - 2
                if len_playlist >= 140:
                    num_140_.append(row[2:])
                elif len_playlist >= 100:
                    num_100_140.append(row[2:])
                elif len_playlist >= 60:
                    num_60_100.append(row[2:])
                elif len_playlist >= 30:
                    num_30_60.append(row[2:])
            if index > end_idx:
                break
        print("num playlists ", num_playlists_more_30)
        print("num_140_ ", len(num_140_))
        print("num_100_140 ", len(num_100_140))
        print("num_60_100 ", len(num_60_100))
        print("num_30_60 ", len(num_30_60))


if __name__ == "__main__":
    create_dataset()
