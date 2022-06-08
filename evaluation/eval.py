"""
Evaluation script.
It considers how good a model recommends relevant tracks (R-Precision) and
it evaluates the ordering of the recommendation (NDCG).
"""
import gensim

from evaluation import load_eval_data as eval_data
import numpy as np
from collections import OrderedDict
import torch

WORD2VEC_TRACKS_PATH = "/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model"
WORD2VEC_ARTISTS_PATH = "/netscratch/kamke/models/word2vec/1_mil_playlists_artists/word2vec-song-vectors.model"


def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device):
    print("start evaluation...")
    # create evaluation dataset
    print("create evaluation dataset...")
    evaluation_dataset = eval_data.SpotifyEvaluationDataset(trackUri2trackId, artistUri2artistId, start_idx, end_idx)
    print("Length of the evaluation dataset: " + str(len(evaluation_dataset)) +
          " (start_idx: " + str(start_idx) + ", end_idx: " + str(end_idx) + ")")
    print("finished")

    # loop over all evaluation playlists
    print("start computing R-Precision and NDCG:")
    r_precision_tracks_sum = 0.0
    r_precision_artists_sum = 0.0
    ndcg_tracks_sum = 0.0
    ndcg_artists_sum = 0.0

    for i, (src, trg, pid, title) in enumerate(evaluation_dataset):
        print("playlist " + str(i) + " of " + str(len(evaluation_dataset)) + " -----------------")
        print("PID = " + str(pid) + ", length playlist: " + str(len(src) + len(trg)))
        # src (list of indices), trg (list of indices)
        src = src.to(device)
        trg = trg.to(device)
        num_predictions = len(trg)
        # num_predictions = 500
        prediction = model.predict(src, num_predictions)

        # prediction is of shape len(trg)
        # first compute R-Precision and NDCG for tracks
        r_precision_tracks = calc_r_precision(prediction, trg)
        ndcg_tracks = calc_ndcg(prediction, trg)
        r_precision_tracks_sum += r_precision_tracks
        ndcg_tracks_sum += ndcg_tracks

        # convert prediction and target to list's of artist id's
        artist_prediction, artist_ground_truth = tracks_to_artists(trackId2artistId, prediction, trg)

        # calculate for the artists R-Precision and artists NDCG
        r_precision_artists = calc_r_precision(artist_prediction, artist_ground_truth)
        ndcg_artists = calc_ndcg(artist_prediction, artist_ground_truth)
        r_precision_artists_sum += r_precision_artists
        ndcg_artists_sum += ndcg_artists

        print("R-Precision(tracks) : " + str(r_precision_tracks))
        print("R-Precision(artists): " + str(r_precision_artists))
        print("NDCG(tracks):       : " + str(ndcg_tracks))
        print("NDCG(artists):      : " + str(ndcg_artists))
        print(" ")

    r_precision_tracks_sum = r_precision_tracks_sum / len(evaluation_dataset)
    ndcg_tracks_sum = ndcg_tracks_sum / len(evaluation_dataset)
    r_precision_artists_sum = r_precision_artists_sum / len(evaluation_dataset)
    ndcg_artists_sum = ndcg_artists_sum / len(evaluation_dataset)
    r_precision = (r_precision_tracks_sum + r_precision_artists_sum) / 2.0
    ndcg = (ndcg_tracks_sum + ndcg_artists_sum) / 2.0

    # print the results
    print("Results for evaluation dataset ----------------------------")
    print("Average R-Precision(tracks) : " + str(r_precision_tracks_sum))
    print("Average R-Precision(artists): " + str(r_precision_artists_sum))
    print("Average NDCG(tracks):       : " + str(ndcg_tracks_sum))
    print("Average NDCG(artists):      : " + str(ndcg_artists_sum))
    print("---> R-Precision            : " + str(r_precision))
    print("---> NDCG                   : " + str(ndcg))

    output_string = "Results for evaluation dataset ----------------------------\n" + \
                    "start_idx: " + str(start_idx) + "\n" \
                                                     "end_idx: " + str(end_idx) + "\n" \
                                                                                  "Average R-Precision(tracks) : " + str(
        r_precision_tracks_sum) + "\n" + \
                    "Average R-Precision(artists): " + str(r_precision_artists_sum) + "\n" + \
                    "Average NDCG(tracks):       : " + str(ndcg_tracks_sum) + "\n" + \
                    "Average NDCG(artists):      : " + str(ndcg_artists_sum) + "\n" + \
                    "---> R-Precision            : " + str(r_precision) + "\n" + \
                    "---> NDCG                   : " + str(ndcg)
    return output_string


def evaluate_title2rec_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device):
    print("start evaluation...")
    # create evaluation dataset
    print("create evaluation dataset...")
    evaluation_dataset = eval_data.SpotifyEvaluationDataset(trackUri2trackId, artistUri2artistId, start_idx, end_idx)
    print("Length of the evaluation dataset: " + str(len(evaluation_dataset)) +
          " (start_idx: " + str(start_idx) + ", end_idx: " + str(end_idx) + ")")
    print("finished")

    # loop over all evaluation playlists
    print("start computing R-Precision and NDCG:")
    r_precision_tracks_sum = 0.0
    r_precision_artists_sum = 0.0
    ndcg_tracks_sum = 0.0
    ndcg_artists_sum = 0.0

    for i, (src, trg, pid, title) in enumerate(evaluation_dataset):
        print("playlist " + str(i) + " of " + str(len(evaluation_dataset)) + " -----------------")
        print("PID = " + str(pid) + ", Title = " + str(title) + ", length playlist: " + str(len(src) + len(trg)))
        # src (list of indices), trg (list of indices)
        src = src.to(device)
        trg = trg.to(device)
        num_predictions = len(trg)
        # num_predictions = 500
        prediction = model.predict(title, num_predictions)

        # prediction is of shape len(trg)
        # first compute R-Precision and NDCG for tracks
        r_precision_tracks = calc_r_precision(prediction, trg)
        ndcg_tracks = calc_ndcg(prediction, trg)
        r_precision_tracks_sum += r_precision_tracks
        ndcg_tracks_sum += ndcg_tracks

        # convert prediction and target to list's of artist id's
        artist_prediction, artist_ground_truth = tracks_to_artists(trackId2artistId, prediction, trg)

        # calculate for the artists R-Precision and artists NDCG
        r_precision_artists = calc_r_precision(artist_prediction, artist_ground_truth)
        ndcg_artists = calc_ndcg(artist_prediction, artist_ground_truth)
        r_precision_artists_sum += r_precision_artists
        ndcg_artists_sum += ndcg_artists

        print("R-Precision(tracks) : " + str(r_precision_tracks))
        print("R-Precision(artists): " + str(r_precision_artists))
        print("NDCG(tracks):       : " + str(ndcg_tracks))
        print("NDCG(artists):      : " + str(ndcg_artists))
        print(" ")

    r_precision_tracks_sum = r_precision_tracks_sum / len(evaluation_dataset)
    ndcg_tracks_sum = ndcg_tracks_sum / len(evaluation_dataset)
    r_precision_artists_sum = r_precision_artists_sum / len(evaluation_dataset)
    ndcg_artists_sum = ndcg_artists_sum / len(evaluation_dataset)
    r_precision = (r_precision_tracks_sum + r_precision_artists_sum) / 2.0
    ndcg = (ndcg_tracks_sum + ndcg_artists_sum) / 2.0

    # print the results
    print("Results for evaluation dataset ----------------------------")
    print("Average R-Precision(tracks) : " + str(r_precision_tracks_sum))
    print("Average R-Precision(artists): " + str(r_precision_artists_sum))
    print("Average NDCG(tracks):       : " + str(ndcg_tracks_sum))
    print("Average NDCG(artists):      : " + str(ndcg_artists_sum))
    print("---> R-Precision            : " + str(r_precision))
    print("---> NDCG                   : " + str(ndcg))

    output_string = "Results for evaluation dataset ----------------------------\n" + \
                    "start_idx: " + str(start_idx) + "\n" \
                                                     "end_idx: " + str(end_idx) + "\n" \
                                                                                  "Average R-Precision(tracks) : " + str(
        r_precision_tracks_sum) + "\n" + \
                    "Average R-Precision(artists): " + str(r_precision_artists_sum) + "\n" + \
                    "Average NDCG(tracks):       : " + str(ndcg_tracks_sum) + "\n" + \
                    "Average NDCG(artists):      : " + str(ndcg_artists_sum) + "\n" + \
                    "---> R-Precision            : " + str(r_precision) + "\n" + \
                    "---> NDCG                   : " + str(ndcg)
    return output_string

# ----------------------------------------------------------------------------------------------------------------------
"""
functions for calculating R-Precision and NDCG
All metrics will be evaluated at both the track level (exact track match)
and the artist level (any track by the same artist is a match)
"""


def calc_r_precision(prediction, ground_truth):
    # rel_tracks = np.intersect1d(prediction, ground_truth)
    num_rel_tracks = 0
    for id in prediction:
        if id in ground_truth:
            num_rel_tracks += 1
    return float(num_rel_tracks) / float(len(ground_truth))


def calc_dcg(prediction, ground_truth):
    unique_predicted = to_unique_tensor(prediction)
    unique_ground_truth = to_unique_tensor(ground_truth)
    if len(unique_predicted) == 0 or len(unique_ground_truth) == 0:
        return 0.0
    score = [float(elem in unique_ground_truth) for elem in unique_predicted]
    return np.sum(score / np.log2(1 + np.arange(1, 1 + len(score))))


def calc_ndcg(prediction, ground_truth):
    dcg = calc_dcg(prediction, ground_truth)
    ideal_dcg = calc_dcg(ground_truth, ground_truth)
    return dcg / ideal_dcg


def tracks_to_artists(artist_dict, prediction, ground_truth):
    artist_pred = prediction
    artist_ground_truth = ground_truth
    for i, track_id in enumerate(prediction):
        artist_pred[i] = artist_dict[int(track_id)]
    for i, track_id in enumerate(ground_truth):
        artist_ground_truth[i] = artist_dict[int(track_id)]
    return artist_pred, artist_ground_truth


def to_unique_tensor(tensor):
    if not isinstance(tensor, list):
        tensor_list = tensor.tolist()
    else:
        tensor_list = tensor
    unique_tensor_list = list(OrderedDict.fromkeys(tensor_list))
    return torch.LongTensor(unique_tensor_list)
