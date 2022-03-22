"""
for converting the mpd to csv tables run: python evaluation/mpd2csv.py --mpd_path /path/to/mpd --out_path dataset
"""

from os import listdir
from os import path
import argparse
import csv
import json

parser = argparse.ArgumentParser(description="Convert MPD(json) to csv file with list of spotify uri's")

parser.add_argument('--mpd_path', default=None, required=True)
parser.add_argument('--out_path', default=None, required=True)

args = parser.parse_args()

playlists_file = open(path.join(args.out_path, 'track_sequences.csv'), 'w', newline='', encoding='utf8')
playlists_writer = csv.writer(playlists_file)

for mpd_slice in listdir(args.mpd_path):
    with open(path.join(args.mpd_path, mpd_slice), encoding='utf8') as json_file:
        print("\tReading file " + mpd_slice)
        json_slice = json.load(json_file)

        for playlist in json_slice['playlists']:
            row = []
            row.extend([playlist['pid'], playlist['name']])
            for track in playlist['tracks']:
                row.append(track['track_uri'])
            playlists_writer.writerow(row)