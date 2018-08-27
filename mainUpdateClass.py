import numpy as np
import scipy.io
import os
import copy
import scipy.io as scio
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score
import time
import csv
from pclassUpdated import *
from utility import *

def main():

    print('Please input IterationTime: ')
    #iteration_time = int(raw_input())
    iteration_time = 200
    folder = 'data2'
    file_list = os.listdir(folder)
    # print(file_list)
    count = 0
    for file_name in file_list:
        print('Processing file: ', file_name)
        result_file = file_name.split('.')[0]
        print('i-th file', count)
        count += 1
        _, data_num, data_coor, relation, x_data_coor, y_data_coor = load_data(folder+os.sep+file_name)
        train_data_ind = np.where(relation < 2)[0]
        train_relation = relation[train_data_ind]
        train_data_coor = data_coor[train_data_ind, :]

        test_data_ind = np.where(relation > 1)[0]
        # Test relation need to be -2 to be 0 or 1
        test_relation = relation[test_data_ind] - 2
        # print('Test relation are : ')
        # print(test_relation)
        print('How many test relations: ', len(test_relation))
        print('How many 1s in test relation: ', np.sum(test_relation))
        test_data_coor = data_coor[test_data_ind, :]

        # print('Initialize hyper parameters')
        max_stage = 100
        beta = 2
        dim_num = 2
        particle_num = 10
        budget = 3
        internal_round = 1
        result_csv_folder= 'csv'+str(internal_round)
        # iteration_time = int(raw_input())

        print('Initialize Tree')
        # (self, beta, maxStage, budget, dimNum)
        # tree = Initialize_currentParticle(beta, maxStage, budget, dimNum, dataNum, relation, dataCoor)
        tree = Tree(beta, max_stage, budget, dim_num, data_num)
        tree.tree_full_gen()
        # print('After tree full gen, number of Cuts: ', len(tree.node_num_seq))
        # print('After tree full gen, nodes are: ', tree.candidate_set[-1])
        # print('After update, last node: ', tree.node_num_seq[-1])
        # tree.visualize()

        train_ll_seq = np.zeros([iteration_time + 1])
        auc_seq = np.zeros([iteration_time + 1])
        trainll = tree.ll_cal(train_data_coor, train_relation, tree.node_num_seq[-1])
        # Calculate every test data belong to which block
        # Calculate how many training points belong to each block
        test_scores = predictTestScore(tree, train_data_coor, train_relation, test_data_coor, test_relation)
        # AUC calculation provided by python,but a little different by testing
        # auc = roc_auc_score(test_relation, test_scores)

        # AUC according to matlab code, but very easy for the situation of divide 0
        auc = calcualteAUC(test_relation, test_scores)
        print('After Inilization, trainll is: ', trainll)
        print('After Inilization, auc is: ', auc)
        train_ll_seq[0] = trainll
        auc_seq[0] = auc
        # loop many times
        for itei in range(iteration_time):
            # if np.remainder(itei, 2) == 0:
            #     print("n-th iteration================", itei, float(itei) / float(iteration_time))
            # print('    Update coors')
            print('Start to update coor i-th: ', itei)
            for j in range(internal_round):
                trainll, auc, data_coor = update_coors_one(tree, trainll, train_data_ind, test_data_ind, train_relation,
                                                           test_relation, data_coor, data_num, beta)
            # train_relation is never changed so no need to update, but train data coor should be updated
            train_data_coor = data_coor[train_data_ind, :]
            # trainll, auc, xDataCoor, yDataCoor = update_coors(tree, x_data_coor, y_data_coor, relation, data_num, beta, relation_matrix)
            train_ll_seq[itei + 1] = trainll
            print('After update coor, Training loglikelihood is: ', trainll)
            print('After update coor, AUC is: ', auc)
            auc_seq[itei + 1] = auc
            current_tree = copy.deepcopy(tree)
            #  divide training data, use training data only to update particles
            # print('Start to update particle i-th: ', itei)
            trainll = tree.update_particle(particle_num, current_tree, max_stage, budget, train_data_coor,
                                           train_relation,
                                           dim_num)
            # print('End of update particle i-th: ', itei)
            # print('After update particle, raining loglikelihood is: ', trainll)
            # print('After update, the number of the CUTS: ', len(tree.node_num_seq))
            #print('After update, Nodes are: ', tree.candidate_set[-1])
            # print('After update, last node: ', tree.node_num_seq[-1])
            # print('After update Tree, Training loglikelihood is: ', trainll)

        if not os.path.exists(result_csv_folder):
            os.makedirs(result_csv_folder)

        ISOTIMEFORMAT = '%m%d%H%M'
        resultFileName = time.strftime(ISOTIMEFORMAT, time.localtime())
        with open(result_csv_folder + os.sep + 'budget' + str(budget) + 'trainll' + resultFileName+result_file + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(train_ll_seq)
        with open(result_csv_folder + os.sep +'budget' + str(budget) + 'AUC' + resultFileName+result_file + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(auc_seq)

        # plt.subplot(121)
        # plt.plot(np.arange(iteration_time + 1), train_ll_seq)
        # plt.subplot(122)
        # plt.plot(np.arange(iteration_time + 1), auc_seq)
        # plt.show()


if __name__ == '__main__':
    main()


