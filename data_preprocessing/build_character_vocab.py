"""
 For playlist titles,
we deal with 41 characters including 26 alphabets (a-z), 10 numbers
(0-9), and 5 special characters (/<>+-)
"""
import csv

PATH_TRACK_SEQUENCES = '/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/track_sequences.csv'


def write_all_titles_in_csv():
    titles_file = open('/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/titles.csv',
                       'w', newline='', encoding='utf8')
    playlists_writer = csv.writer(titles_file)
    with open(PATH_TRACK_SEQUENCES, encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create lists of track uri's
        for row in csv_reader:
            title_str = row[1]
            playlists_writer.writerow(title_str)


if __name__ == "__main__":
    write_all_titles_in_csv()
