import numpy as np
import os
from time import time
import json
import Representation
import threading
import copy
from TimeSeries import TimeSeries
from copy import deepcopy
from Causal_inference import check_with_original
from Causal_Test import general_test_non_GRAIL
import csv

CWD = os.getcwd()
DATASET_NAMES = ["FaceFour","InlineSkate", "PickupGestureWiimoteZ", "SemgHandMovementCh2"]
#"FaceFour", 
TO_IMPORT =  ["mixsd0.1_0.1_causaldb", "mixsd0.1_0.05_causaldb", "mixsd0.2_0.1_causaldb", "mixsd0.2_0.05_causaldb", "randomsd0.1_effectdb", "randomsd0.2_effectdb", "rwalksd0.1_effectdb", "rwalksd0.05_effectdb"]

DATA_PATH = CWD +"/test_files/"

BEST_GAMMA = 5
NEIGHBORS =[2, 5, 10, 100]
PVALS = [0.01, 0.025, 0.05, 0.1]
LAGS = [1, 2]

TAU_MAX = 3

class DataAttri:
  def __init__(self):
    self.dataset_name = None
    self.import_type = None
    self.alpha_level = None
    self.p_value = None
    self.lagged = None
    self.representation = None
    # self.f_score = None
    # self.return_time = None
    # self.trueMat = None
    # self.pcmci = None
    # self.val_matrix = None
    # self.link_matrix = None
    # self.p_matrix = None
    # self.q_matrix = None
    # self.compare_matrix = None
    # self.var_names = None
    self.model = None
    self.shape = None

def import_data():
    dataset_dict = {}
    for each in DATASET_NAMES:
        dataset_dict[each] = {}
        dataset_dict[each]["truemat"] = np.load(str(DATA_PATH+each+"_split_truemat.npy"))
        # print(dataset_dict[each]["truemat"])
        dataset_dict[each]["causaldb"] = np.load(str(DATA_PATH+ each+"_causaldb.npy"))
        for import_type in TO_IMPORT:
            dataset_dict[each][import_type] = np.load(str(DATA_PATH+each+"_"+import_type+".npy"))
    return dataset_dict

def run_test(dataset_dict):
    for each in DATASET_NAMES:
        attr_hold = DataAttri()
        causaldb = dataset_dict[each]["causaldb"]
        # representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = BEST_GAMMA)
        trueMat = dataset_dict[each]["truemat"]
        # attr_hold.trueMat = trueMat

        for import_type in TO_IMPORT:
            attr_hold.dataset_name = each
            attr_hold.import_type = import_type
            effectdb = dataset_dict[each][import_type]
            n1 = causaldb.shape[0]
            n2 = effectdb.shape[0]
            n = n1+n2
            for alpha_level in PVALS:
                for lagged in LAGS:
                    brute_results, result_by_neighbor = general_test_non_GRAIL(causaldb, effectdb, trueMat, best_gamma = BEST_GAMMA, neighbor_param= NEIGHBORS,lag = lagged, pval=alpha_level)
                    attrz = copy.deepcopy(attr_hold)
                    attrz.alpha_level = alpha_level
                    attrz.lagged = lagged
                    attrz.representation = "brute+GRAIL"
                    attrz.model = "granger_general_test"
                    attrz.shape = n
                    y = threading.Thread(target=saving_csv, args=(brute_results, result_by_neighbor, attrz,))
                    y.start()


def saving_csv(brute_results, result_by_neighbor, attr):
    # print(type(attr.shape))
    # print(type(attr.lagged))
    counter = 0
    with open(f'{CWD}/model_results_granger_non_GRAIL/{attr.dataset_name}_{attr.import_type}_P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_results.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["shape"], ["lagged"], ["type"], ["precision"], ["recall"], ["fscore"], ["runtime"])
        csvwriter.writerow([int(attr.shape)] + [int(attr.lagged)] + ['brute'] + list(brute_results.values()))
        csvwriter.writerow(["shape"], ["lagged"], ["neighbors"], ["precision"], ["recall"], ["fscore"], ["runtime"], ["map"], ["knn_recall"])

        for n_num in result_by_neighbor:
                csvwriter.writerow([int(attr.shape)] + [int(attr.lagged)] + [NEIGHBORS[counter]] + list(result_by_neighbor[n_num].values()))
                counter += 1
def main():
    dataset_dict = import_data()
    run_test(dataset_dict)
    pass

if __name__ == "__main__":
    main()