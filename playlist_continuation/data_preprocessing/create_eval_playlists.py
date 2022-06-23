import random
import csv
from playlist_continuation.config import load_attributes as la
from playlist_continuation.data_preprocessing import build_character_vocab as cv

# training playlists:    0 - 980000
# evaluation playlists:  980001 - 999999
# --> evaluation_set contains of 20000 playlists
start_idx = 980001
end_idx = 999999


# here are 13934 playlists, with length >= 30 tracks

def create_dataset():
    num_140_ = []
    num_100_140 = []
    num_60_100 = []
    num_30_60 = []
    with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create lists of track uri's

        """c_title_only = 0
        c_first_5 = 0
        c_first_10 = 0
        c_first_25 = 0
        c_first_100 = 0"""

        num_playlists_more_30 = 0
        for index, row in enumerate(csv_reader):
            if start_idx <= index <= end_idx and len(row) >= 32:
                num_playlists_more_30 += 1
                len_playlist = len(row) - 2
                if len_playlist >= 140:
                    num_140_.append(row)
                elif len_playlist >= 100:
                    num_100_140.append(row)
                elif len_playlist >= 60:
                    num_60_100.append(row)
                elif len_playlist >= 30:
                    num_30_60.append(row)
            if index > end_idx:
                break

        print("num playlists ", num_playlists_more_30)
        print("num_140_ ", len(num_140_))
        print("num_100_140 ", len(num_100_140))
        print("num_60_100 ", len(num_60_100))
        print("num_30_60 ", len(num_30_60))

        first_100 = num_140_[0:2000]
        title_only = []
        for playlist in num_30_60:
            if len(title_only) >= 2000:
                break
            else:
                if len(cv.title2index_seq(playlist[1])) >= 1:
                    title_only.append(row)

        first_25 = []
        first_10 = []
        first_5 = []
        first_1 = []

        for _ in range(2000):
            r = random.uniform(0, 1)
            if r < 0.33:
                first_5.append(num_60_100[0])
                num_60_100 = num_60_100[1:]
            elif 0.33 <= r < 0.66:
                first_5.append(num_30_60[0])
                num_30_60 = num_30_60[1:]
            else:
                first_5.append(num_100_140[0])
                num_100_140 = num_100_140[1:]

        for _ in range(2000):
            r = random.uniform(0, 1)
            if r < 0.33:
                first_10.append(num_60_100[0])
                num_60_100 = num_60_100[1:]
            elif 0.33 <= r < 0.66:
                first_10.append(num_30_60[0])
                num_30_60 = num_30_60[1:]
            else:
                first_10.append(num_100_140[0])
                num_100_140 = num_100_140[1:]

        for _ in range(2000):
            r = random.uniform(0, 1)
            if r < 0.33:
                first_25.append(num_60_100[0])
                num_60_100 = num_60_100[1:]
            elif 0.33 <= r < 0.66:
                first_25.append(num_30_60[0])
                num_30_60 = num_30_60[1:]
            else:
                first_25.append(num_100_140[0])
                num_100_140 = num_100_140[1:]

        for _ in range(2000):
            r = random.uniform(0, 1)
            if r < 0.5:
                first_1.append(num_60_100[0])
                num_60_100 = num_60_100[1:]
            else:
                first_1.append(num_30_60[0])
                num_30_60 = num_30_60[1:]

        print("title_only ", len(title_only))
        print("first_1 ", len(first_1))
        print("first_5 ", len(first_5))
        print("first_10 ", len(first_10))
        print("first_25 ", len(first_25))
        print("first_100 ", len(first_100))

        eval_data = title_only + first_1 + first_5 + first_10 + first_25 + first_100
        print("len(eval_data) = ", len(eval_data))

        with open('/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/eval_data.csv', 'w', newline='',
                  encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            for playlist in eval_data:
                writer.writerow(playlist)

        print("finished")

        """
        num playlists        13935
        len(num_140_)        2336
        len(num_100_140)     2104
        len(num_60_100)      3898
        len(num_30_60)       5597
        in num_30_60 has title 5597
        """
        """
        Create evaluation dataset:
        2000x title only
        2000x first_1
        2000x first_5
        2000x first_10
        2000x first_25
        2000x first_100
        """


if __name__ == "__main__":
    create_dataset()
