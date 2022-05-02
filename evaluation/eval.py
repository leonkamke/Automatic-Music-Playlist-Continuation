"""
EVALUATION OF THE SPOTIFY CHALLENGE:
  1000 examples for each scenario:
  1)  Title only (no tracks)
  2)  Title and first track
  3)  Title and first 5 tracks
  4)  First 5 tracks only
  5)  Title and first 10 tracks
  6)  First 10 tracks only
  7)  Title and first 25 tracks
  8)  Title and 25 random tracks
  9)  Title and first 100 tracks
  10) Title and 100 random tracks
"""

from evaluation import load_eval_data as eval_data
import numpy as np


def evaluate_model(model, word2vec_tracks, word2vec_artists, end_idx):
    print("start evaluation...")
    # create evaluation dataset
    print("create evaluation dataset...")
    evaluation_dataset = eval_data.EvaluationDataset(word2vec_tracks, word2vec_artists, end_idx)
    print("finished")
    # loop over all evaluation playlists
    print("start computing R-Precision and NDCG:")
    r_precision_tracks_sum = 0.0
    r_precision_artists_sum = 0.0
    ndcg_tracks_sum = 0.0
    ndcg_artists_sum = 0.0
    print(len(evaluation_dataset))
    for i, (src, trg) in enumerate(evaluation_dataset):
        print("playlist " + str(i) + " of " + str(len(evaluation_dataset)))
        # src (list of indices), trg (list of indices), trg_len (natural number)
        prediction = model.predict(src)
        print(prediction)
        # prediction is of shape len(trg)
        # first compute R-Precision and NDCG for tracks
        r_precision_tracks = calc_r_precision(prediction, trg)
        ndcg_tracks = calc_NDCG(prediction, trg)
        r_precision_tracks_sum += r_precision_tracks
        ndcg_tracks_sum += ndcg_tracks
        # convert prediction and target to list's of artist id's
        artist_prediction, artist_ground_truth = tracks_to_artists(evaluation_dataset.artist_dict, prediction, trg)
        # calculate for the artists R-Precision and NDCG
        r_precision_artists = calc_r_precision(artist_prediction, artist_ground_truth)
        ndcg_artists = calc_NDCG(artist_prediction, artist_ground_truth)
        r_precision_artists_sum += r_precision_artists
        ndcg_artists_sum += ndcg_artists

        print("R-Precision(tracks) : " + str(r_precision_tracks))
        print("R-Precision(artists): " + str(r_precision_artists))
        print("NDCG(tracks):       : " + str(ndcg_tracks))
        print("NDCG(artists):      : " + str(ndcg_artists))
        print(" ")

    r_precision_tracks = r_precision_tracks_sum / len(evaluation_dataset)
    ndcg_tracks = ndcg_tracks_sum / len(evaluation_dataset)
    r_precision_artists = r_precision_artists_sum / len(evaluation_dataset)
    ndcg_artists = ndcg_artists_sum / len(evaluation_dataset)
    # print the results
    print("Average R-Precision(tracks) : " + str(r_precision_tracks))
    print("Average R-Precision(artists): " + str(r_precision_artists))
    print("Average NDCG(tracks):       : " + str(ndcg_tracks))
    print("Average NDCG(artists):      : " + str(ndcg_artists))


# ----------------------------------------------------------------------------------------------------------------------
# functions for calculating R-Precision, NDCG
# All metrics will be evaluated at both the track level (exact track match)
# and the artist level (any track by the same artist is a match)


def calc_r_precision(prediction, ground_truth):
    rel_tracks = np.intersect1d(prediction, ground_truth)
    return len(rel_tracks) / len(ground_truth)


def calc_NDCG(prediction, ground_truth):
    seq_len = len(prediction)
    relevance = np.arange(seq_len, 0, -1)   # list goes from seq_len to 1
    rel_dict = {}
    # create dictionary
    for i, rel_i in enumerate(relevance):
        element = ground_truth[i]
        rel_dict[element] = rel_i
    # compute ndcg with dcg and ideal dcg
    dcg = 0.0
    for i, elem in enumerate(prediction):
        # compute dcg
        if elem in rel_dict:
            dcg += rel_dict[elem] / np.log2(i+2)
    # compute idcg
    idcg = 0.0
    for i, rel in enumerate(relevance):
        idcg += rel / np.log2(i+2)
    ndcg = dcg / idcg
    return ndcg


def tracks_to_artists(artist_dict, prediction, ground_truth):
    artist_pred = prediction
    artist_ground_truth = ground_truth
    for i, track_id in enumerate(prediction):
        artist_pred[i] = artist_dict[int(track_id)]
    for i, track_id in enumerate(ground_truth):
        artist_ground_truth[i] = artist_dict[int(track_id)]
    return artist_pred, artist_ground_truth

