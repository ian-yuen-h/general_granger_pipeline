import Representation
import numpy as np
from kNN import kNN, kNN_with_pq_NCC, kNN_with_pq_SINK, MAP, avg_recall_measure
from time import time
from Causal_inference import generate_synthetic, granger_matrix, check_with_original, granger_causality, general_granger_matrix
import csv


best_gamma = 5

#load the fixed datasets
def load_ts_truemat(lag, n, ar = 'ar05'):
    with open('./datasets_'+ar+'/series'+str(lag)+'_'+str(n)+'.npy', 'rb') as f:
        TS = np.load(f)
    with open('./datasets_' + ar + '/truemat' + str(lag) + '_' + str(n) + '.npy', 'rb') as f:
        trueMat = np.load(f)
    return TS, trueMat



def general_test(Cause_TS, Effect_TS, trueMat, best_gamma, neighbor_param =[2, 5, 10, 100], lag = 2, pval = 0.05):
    """
    test() but you can specify query database and effect database
    :param n: Number of time series
    :param lag: The lag of causality
    :param m: time series size
    :return: brute_results is the results of the standard method,
    results_by_neighbor is the results with GRAiL Pruning
    """
    n1 = Cause_TS.shape[0]
    n2 = Effect_TS.shape[0]
    representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = best_gamma)

    grailMat = np.zeros((n1, n2))
    t = time()
    bruteMat = general_granger_matrix(Cause_TS,Effect_TS, lag,pval=pval)
    bruteTime = time() - t

    result_by_neighbor = {}

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(Effect_TS, Cause_TS)
    brute_res = check_with_original(trueMat, bruteMat)
    brute_results = {'precision' : brute_res[0], 'recall' : brute_res[1],
                                            'fscore' : brute_res[2], 'runtime' : bruteTime}

    for neighbor_num in neighbor_param:
        if neighbor_num >= n2:
            continue
        neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=neighbor_num, representation=None, use_exact_rep=True,
                              pq_method=None) #changed pq

        exact_neighbors, _, _ = kNN(Effect_TS, Cause_TS, method="SINK", k=neighbor_num, representation=None, gamma_val=best_gamma)

        knn_map_accuracy = MAP(exact_neighbors, neighbors)
        knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)

        print(knn_recall_accuracy, knn_map_accuracy)

        t = time()
        for i in range(n1):
           for j in neighbors[i]:
                grailMat[i,j] = granger_causality(Effect_TS[j], Cause_TS[i], lag, pval=pval)
        prunedtime = time() - t

        grail_results = check_with_original(trueMat, grailMat)
        result_by_neighbor[neighbor_num] = {'precision' : grail_results[0], 'recall' : grail_results[1],
                                            'fscore' : grail_results[2], 'runtime' : prunedtime,'map' : knn_map_accuracy,
                                            'knn_recall' : knn_recall_accuracy}

    return brute_results, result_by_neighbor


def test_only_grail(TS, trueMat, n = 100, lag = 2, m = 128, neighbor_param = [10,100], tune = False):
    #try not setting gamma
    d = int(min(max(np.ceil(0.4 * 2*n), 20), 100))
    if tune:
        representation = Representation.GRAIL(kernel="SINK", d=100)
    else:
        representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = best_gamma)

    grailMat = np.zeros((n, n))

    result_by_neighbor = {}

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)

    for neighbor_num in neighbor_param:
        if neighbor_num >= n:
            continue
        np.random.seed(0)
        neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=neighbor_num, representation=None, use_exact_rep=True,
                              pq_method=None)

        exact_neighbors, _, _ = kNN(TS, TS, method="SINK", k=neighbor_num, representation=None, gamma_val=best_gamma)

        knn_map_accuracy = MAP(exact_neighbors, neighbors)
        knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)

        t = time()
        for i in range(n):
           for j in neighbors[i]:
               if j != i:
                  grailMat[i,j] = granger_causality(TS[j], TS[i], lag)
        prunedtime = time() - t

        grail_results = check_with_original(trueMat, grailMat)
        result_by_neighbor[neighbor_num] = {'precision' : grail_results[0], 'recall' : grail_results[1],
                                            'fscore' : grail_results[2], 'runtime' : prunedtime, 'map' : knn_map_accuracy,
                                            'knn_recall' : knn_recall_accuracy}

    return result_by_neighbor


def test(TS, trueMat, best_gamma, neighbor_param =[2, 5, 10, 100], lag = 2, pval = 0.05):
    """
    Perform tests of accuracy and time on GRAIL and standard granger causality tests.
    :param n: Number of time series
    :param lag: The lag of causality
    :param m: time series size
    :return: brute_results is the results of the standard method,
    results_by_neighbor is the results with GRAiL Pruning
    """
    n,m = TS.shape
    #try not setting gamma
    #d = int(min(max(np.ceil(0.4 * 2*n), 20), 100))
    representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = best_gamma)

    grailMat = np.zeros((n, n))
    controlMat = np.zeros((n, n))
    t = time()
    bruteMat = granger_matrix(TS, lag,pval=pval)
    bruteTime = time() - t

    result_by_neighbor = {}

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)
    brute_res = check_with_original(trueMat, bruteMat)
    brute_results = {'precision' : brute_res[0], 'recall' : brute_res[1],
                                            'fscore' : brute_res[2], 'runtime' : bruteTime}

    for neighbor_num in neighbor_param:
        if neighbor_num >= n:
            continue
    #    neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method = "ED", k = neighbor_num, representation=None, use_exact_rep = True, pq_method = "pq", M = 16)
        #np.random.seed(0)
        neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=neighbor_num, representation=None, use_exact_rep=True,
                              pq_method=None) #changed pq

        exact_neighbors, _, _ = kNN(TS, TS, method="SINK", k=neighbor_num, representation=None, gamma_val=best_gamma)

        knn_map_accuracy = MAP(exact_neighbors, neighbors)
        knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)

        print(knn_recall_accuracy, knn_map_accuracy)

        # control_neighbors = np.array([np.random.choice(n, neighbor_num, replace= False) for i in range(n)])
        # print(control_neighbors)
        #neighbors, _, _ = kNN_with_pq_NCC(TS, TS, k = neighbor_num, use_exact_rep=True, Ks=4, M = 16)
        #neighbors, _, _ = kNN(TS, TS, method = "NCC", k = neighbor_num, representation=None, use_exact_rep = True, pq_method = None, M = 16)

        #neighbors, _, _ = kNN_with_pq_SINK(TS, TS, k = neighbor_num, use_exact_rep=True, Ks=4, M = 16, gamma_val = 1)

        t = time()
        for i in range(n):
           for j in neighbors[i]:
               if j != i:
                  grailMat[i,j] = granger_causality(TS[j], TS[i], lag, pval=pval)
        prunedtime = time() - t

        grail_results = check_with_original(trueMat, grailMat)

        #print("k number for kNN:", neighbor_num)
        #print("Pruned VL time: ", prunedtime)
        #print("prec, rec, and F1 for GRAIL: ", compare_results)
        # print("prec, rec, and F1 for control: ", check_with_original(trueMat, controlMat))
        #print("prec, rec, and F1 for Brute Granger Causality:", brute_results)
        result_by_neighbor[neighbor_num] = {'precision' : grail_results[0], 'recall' : grail_results[1],
                                            'fscore' : grail_results[2], 'runtime' : prunedtime,'map' : knn_map_accuracy,
                                            'knn_recall' : knn_recall_accuracy}

    #print("time for VL:", bruteTime )
    return brute_results, result_by_neighbor


def test_brute_truth(TS, trueMat, best_gamma, neighbor_param =[2, 5, 10, 100], lag = 2, pval = 0.05):
    n,m = TS.shape
    representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = best_gamma)

    grailMat = np.zeros((n, n))

    result_by_neighbor = {}

    TRAIN_TS, TEST_TS = representation.get_rep_train_test(TS, TS)
    brute_res = check_with_original(trueMat, trueMat)
    brute_results = {'precision' : brute_res[0], 'recall' : brute_res[1],
                                            'fscore' : brute_res[2]}

    for neighbor_num in neighbor_param:
        if neighbor_num >= n:
            continue
        neighbors, _, _ = kNN(TRAIN_TS, TEST_TS, method="ED", k=neighbor_num, representation=None, use_exact_rep=True,
                              pq_method=None) #changed pq

        exact_neighbors, _, _ = kNN(TS, TS, method="SINK", k=neighbor_num, representation=None, gamma_val=best_gamma)

        knn_map_accuracy = MAP(exact_neighbors, neighbors)
        knn_recall_accuracy = avg_recall_measure(exact_neighbors, neighbors)

        print(knn_recall_accuracy, knn_map_accuracy)

        t = time()
        for i in range(n):
           for j in neighbors[i]:
               if j != i:
                  grailMat[i,j] = granger_causality(TS[j], TS[i], lag, pval=pval)
        prunedtime = time() - t

        grail_results = check_with_original(trueMat, grailMat)
        result_by_neighbor[neighbor_num] = {'precision' : grail_results[0], 'recall' : grail_results[1],
                                            'fscore' : grail_results[2], 'runtime' : prunedtime,'map' : knn_map_accuracy,
                                            'knn_recall' : knn_recall_accuracy}

    return brute_results, result_by_neighbor


def compare_with_standard(csvname = 'causal_inference_fix.csv'):
    """
    Compare Grail pruning with standard granger causality method
    :return:
    """
    csvfile = open(csvname, 'w')
    csvwriter = csv.writer(csvfile)
    m =128
    for n in range(100, 5000, 100):
        for lag in [2,5,10]:
            print(n, lag)
            TS, trueMat = load_ts_truemat(lag, n, ar='ar05')
            brute_results, result_by_neighbor= test(TS, trueMat, n,lag,m)
            csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
            for n_num in result_by_neighbor:
                csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))

            csvfile.flush()
    csvfile.close()

def scale_grail():
    """
    Show that granger causality with GRAIL can scale
    :return:
    """
    csvfile = open('scale_grail_fix.csv', 'w')
    csvwriter = csv.writer(csvfile)
    m =128
    lags = [2,5,10]

    for n in range(200, 10000000, 500):
        for lag in lags:
            print(n, lag)
            TS, trueMat = load_ts_truemat(lag, n, ar='ar05')
            result_by_neighbor= test_only_grail(TS, trueMat, n,lag,m)
            for n_num in result_by_neighbor:
                csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))

            csvfile.flush()
    csvfile.close()

def deneme():
    TS = np.load('datasets_ar4/series2_200.npy')
    trueMat = np.load('datasets_ar4/truemat2_200.npy')
    n = TS.shape[0]
    lag = 2
    print(trueMat.shape)

    csvfile = open('ar4.csv', 'w')
    csvwriter = csv.writer(csvfile)
    brute_results, result_by_neighbor = test(TS, trueMat, best_gamma = best_gamma, neighbor_param= [10, 100],lag = lag, pval=1e-5)
    csvwriter.writerow([n] + [lag] + ['brute'] + list(brute_results.values()))
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))
    csvfile.close()


if __name__ == '__main__':
    #scale_grail()
    deneme()










