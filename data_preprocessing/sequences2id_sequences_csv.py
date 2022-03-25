from os import listdir
from os import path
import argparse
import csv
import json

def create_dict():
    with open('../data/spotify_million_playlist_dataset_csv/data/vocabulary.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        track_to_index = {}
        index_to_track = {}
        for index, row in enumerate(csv_reader):
            track_to_index[row[1]] = int(row[0])
            index_to_track[row[0]] = row[1]
        return track_to_index, index_to_track


if __name__ == '__main__':
    track_to_index, index_to_track = create_dict()

    file = open('../data/spotify_million_playlist_dataset_csv/data/id_sequences.csv', 'w', newline='', encoding='utf8')
    file_writer = csv.writer(file)

    with open('../data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for row in csv_reader:
            print(row[0])
            to_write = []
            if len(row) >= 3:
                for i in range(2, len(row)):
                    spotify_uri = row[i]
                    to_write.append(track_to_index[spotify_uri])
                file_writer.writerow(to_write)


