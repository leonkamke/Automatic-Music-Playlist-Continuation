import torch
import csv
from playlist_continuation.config import load_attributes as la

# training playlists:    0 - 980000
# evaluation playlists:  980001 - 999999
start_idx = 980001
end_idx = 999999

def create_dataset():
    with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create lists of track uri's

        c_title_only = 0
        c_first_5 = 0
        c_first_10 = 0
        c_first_25 = 0
        c_first_100 = 0
        num_playlists_more_30 = 0
        for index, row in enumerate(csv_reader):
            if start_idx <= index < end_idx and len(row) >= 32:
                num_playlists_more_30 += 1
                """is_odd = len(row) % 2 == 1
                i = int(len(row) / 2 + 1)
                pids.append(row[0])
                src_i = row[2:i]
                trg_i = row[i:len(row)]
                if is_odd:
                    trg_i = row[i:len(row) - 1]
                src_uri.append(src_i)
                trg_uri.append(trg_i)
                titles.append(row[1])"""
            if index > end_idx:
                break
        print(num_playlists_more_30)


if __name__ == "__main__":
    create_dataset()
