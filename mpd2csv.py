"""
for converting the mpd to csv tables run: python evaluation/mpd2csv.py --mpd_path /path/to/mpd --out_path dataset
"""

from os import listdir
from os import path
import argparse
import csv
import json

parser = argparse.ArgumentParser(description="Convert MPD(json) to csv files")

parser.add_argument('--mpd_path', default=None, required=True)
parser.add_argument('--out_path', default=None, required=True)

args = parser.parse_args()

items_file = open(path.join(args.out_path, 'items.csv'), 'w', newline='', encoding='utf8')
items_writer = csv.writer(items_file)

playlists_file = open(path.join(args.out_path, 'playlists.csv'), 'w', newline='', encoding='utf8')
playlists_writer = csv.writer(playlists_file)

tracks_file = open(path.join(args.out_path, 'tracks.csv'), 'w', newline='', encoding='utf8')
tracks_writer = csv.writer(tracks_file)

playlists_descr_file = open(path.join(args.out_path, 'playlists_descr.csv'), 'w', newline='', encoding='utf8')
playlists_descr_writer = csv.writer(playlists_descr_file)

tracks = set()

for mpd_slice in listdir(args.mpd_path):
    with open(path.join(args.mpd_path, mpd_slice), encoding='utf8') as json_file:
        print("\tReading file " + mpd_slice)
        json_slice = json.load(json_file)

        for playlist in json_slice['playlists']:
            playlists_writer.writerow([playlist['pid'], playlist['name'],
                                       playlist['collaborative'], playlist['num_tracks'],
                                       playlist['num_artists'], playlist['num_albums'],

                                       playlist['num_followers'], playlist['num_edits'],
                                       playlist['modified_at'], playlist['duration_ms']])

            if 'description' in playlist:
                playlists_writer.writerow([playlist['pid'], playlist['description']])

            for track in playlist['tracks']:
                items_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

                if track['track_uri'] not in tracks:
                    # This is a new track
                    tracks.add(track['track_uri'])
                    tracks_writer.writerow([track['track_uri'], track['track_name'],
                                            track['artist_uri'], track['artist_name'],
                                            track['album_uri'], track['album_name'],
                                            track['duration_ms']])