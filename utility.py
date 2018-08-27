import numpy as np
# import autograd.numpy as np
import scipy.io as scio
from scipy.stats import t
# import matplotlib.pyplot as plt
import pandas as pd
import copy
import scipy
# from sklearn.metrics import roc_auc_score


# import autograd as grad

# global likelihood_setting
def predictTestScore(tree, train_data_coor, train_relation, test_data_coor, test_relation):
    training_data_belonging = tree.dts_coor_belong(train_data_coor, tree.node_num_seq[-1])
    # calculate how many data belong to each leaf nodes
    # calculate the eta of each leaf node
    # calculate test data belong to which leaf nodes
    # return the corresponding score
    # kCluster is all the leaf nodes contains training data
    kCluster, ind = np.unique(training_data_belonging, return_inverse=True)
    lenKCluster = len(kCluster)
    m1 = np.zeros(lenKCluster)
    m0 = np.zeros(lenKCluster)
    mSum = np.zeros(lenKCluster)
    test_data_num = len(test_relation)
    predicted_scores = np.zeros(test_data_num)

    # Can this for loop be optimized?
    for k in range(lenKCluster):
        m = np.where(np.array(training_data_belonging) == kCluster[k])[0]
        # m = np.where(terminal_node ==kCluster[k])
        # print("kCluster:  ----", k, "totalValue", m)
        # mSum[k] = len(m[0])
        mSum[k] = len(m)
        m0[k] = len(np.where(train_relation[m] == 0)[0])
        m1[k] = len(np.where(train_relation[m] == 1)[0])
    # m0 = mSum - m1
    mtest = mSum - m0 - m1
    if np.sum(mtest) != 0:
        print('-------------ERROR--------------')
        print('-------------ERROR--------------')
        return 'error stop here'
    test_data_belonging = tree.dts_coor_belong(test_data_coor, tree.node_num_seq[-1])

    # predicted_scores =0
    for n in range(test_data_num):
        node = test_data_belonging[n]
        k = np.where(kCluster == node)[0]
        if len(k) == 0:
            predicted_scores[n] = float(tree.beta) / (2 * tree.beta)
        else:
            predicted_scores[n] = float(m1[k] + tree.beta) / (mSum[k] + 2 * tree.beta)
        # if len(k)!=1:
            # print("kCluster is: ")
            # print(kCluster)
            # print('Node is: ')
            # print(node)
            # print('k is: ')
            # print(k)
            # print('----------ERROR happend here!------------')
    return predicted_scores

def calcualteAUC(y_true, y_scores):
    n1 = np.sum(y_true)
    no = len(y_true) - n1
    rank_indcs =np.argsort(y_scores)
    R_sorted = y_true[rank_indcs]
    #+1 because indices in matlab begins with 1
    # #however in python, begins with 0
    So=np.sum(np.where(R_sorted>0)[0]+1)
    aucValue = float(So - (n1*(n1+1))/2)/(n1*no)
    return aucValue


def load_data(fileName):
    # Initialize the Coordinates of each point to [dataNum**2,2] matrix
    # Initialization relation to [dataNum**2, 1] matrix
    #  print('Please input the data file Name:')
    #  dataFile = raw_input()
    data_file = fileName
    relation_matrix = scio.loadmat(data_file)['datas']
    data_num = int(relation_matrix.shape[1])
    x_data_coor = np.random.uniform(0, 1, data_num)
    y_data_coor = np.random.uniform(0, 1, data_num)
    relation_num = data_num * data_num
    # relationvector = np.zeros(relation_num)
    data_coor = np.zeros([relation_num, 2])
    data_coor[:, 0] = np.repeat(x_data_coor, data_num)
    data_coor[:, 1] = np.tile(y_data_coor, data_num)
    relationtest = relation_matrix.flatten()
    # testType  =type(relationtest)

    # print(type(relationtest))
    #relation = relation_matrix.flatten().reshape([data_num*data_num, 1])
    # print('---end of loading data--')
    #  print('Relation Matrix Shape is: ', relation.shape)
    #  print(relation)
    return relation_matrix, data_num, data_coor, relationtest, x_data_coor, y_data_coor


def update_coors_one(current_tree, old_loglikelihood, train_data_ind, test_data_ind, train_relation, test_relation, data_coor, dataNum, beta):
    # print('In updating coors, Nodes of tree are: ', current_tree.candidate_set[-1])
    new_x_coordinates = np.random.uniform(0, 1, dataNum)
    new_y_coordinates = np.random.uniform(0, 1, dataNum)

    ber = np.random.uniform(0, 1, dataNum)
    for x in range(dataNum):
        # Old wrong code:
        old_coor_x = data_coor[x * dataNum, 0]
        # old_coor_x = data_coor[x, 0]
        # data_coor[:, 0] = np.repeat(x_data_coor, dataNum)
        data_coor[x * dataNum:(x + 1) * dataNum, 0] = new_x_coordinates[x]
        train_data_coor = data_coor[train_data_ind, :]
        new_loglikelihood = current_tree.ll_cal(train_data_coor, train_relation, current_tree.node_num_seq[-1])
        ratio = new_loglikelihood - old_loglikelihood
        if (np.log(ber[x]) < ratio):
            # Accept
            old_loglikelihood = new_loglikelihood
        else:
            data_coor[x * dataNum:(x + 1) * dataNum, 0] = old_coor_x

    yber = np.random.uniform(0, 1, dataNum)
    for y in range(dataNum):
        # old_coor_y = data_coor[y*dataNum, 1]
        old_coor_y = data_coor[y, 1]
        data_coor[y::dataNum, 1] = new_y_coordinates[y]
        # trainDataCoor = data[trainDataInd]
        # trainDataCoor[:,0] = data[trainDataIndx,0]
        # trainDataCoor[:,1] = data[trainDataIndy,1]
        train_data_coor = data_coor[train_data_ind, :]

        new_loglikelihood = current_tree.ll_cal(train_data_coor, train_relation, current_tree.node_num_seq[-1])
        ratio = new_loglikelihood - old_loglikelihood
        if (np.log(yber[y]) < ratio):
            # Accept
            old_loglikelihood = new_loglikelihood
        else:
            # reject
            data_coor[y::dataNum, 1] = old_coor_y

    train_data_coor = data_coor[train_data_ind, :]
    test_data_coor = data_coor[test_data_ind, :]

    test_scores = predictTestScore(current_tree, train_data_coor, train_relation, test_data_coor, test_relation)
    auc = calcualteAUC(test_relation, test_scores)
    #auc = roc_auc_score(test_relation, test_scores)

    return old_loglikelihood, auc, data_coor


def update_coors(tree, xDataCoor, yDataCoor, relation, dataNum, beta, relationMatrix):
    beta = tree.beta

    z_label = tree.z_label
    prop_z = z_label.flatten()
    leafNodes = tree.candidate_set[-1]

    numOfClass = len(leafNodes)
    # Calculate how many 1 and 0s in each leaf nodes
    tau1_k = np.zeros(numOfClass)
    tau0_k = np.zeros(numOfClass)
    auc = 0.5
    trainll = 0

    # i=0

    for i in range(numOfClass):
        pointIdBelongToNode = np.where(prop_z == leafNodes[i])[0]
        relationBelongToNode = relation[pointIdBelongToNode]
        tau1_k[i] = np.sum(np.where(relationBelongToNode == 1)[0])
        tau0_k[i] = np.sum(np.where(relationBelongToNode == 0)[0])

    if numOfClass == 1:
        # Will update trainll later when finish the main program
        tau_k = tau0_k + tau1_k
        probs = (tau1_k + beta) / (tau_k + 2 * beta)
        trainll = np.sum(np.where(relation == 1)[0]) * np.log(probs) + np.sum(np.where(relation == 0)[0]) * np.log(
            1 - probs)

        return (trainll, auc, xDataCoor, yDataCoor)

    newXCoordinates = np.random.uniform(0, 1, dataNum)
    newYCoordinates = np.random.uniform(0, 1, dataNum)
    rand = np.random.uniform(0, 1, [dataNum, 2])
    z_label = tree.z_label
    for x in range(dataNum):
        # Calculate for the current x, how may belong to each node with relation 0 or 1
        # To realise this sentence:
        # origin_ii1 = hist((z_label(ii, datas(ii, :)==1)), 1:numClass);

        # Here I used for loop to calculate hist, which can be optimized later
        # May need to minus z_label by one

        # xRelatedRelation = relationMatrix[x,:]

        xRelatedLabel = z_label[x, :]
        origin_ii1 = np.zeros(numOfClass)
        origin_ii0 = np.zeros(numOfClass)
        for k in range(numOfClass):
            pIdBTN = np.where(xRelatedLabel == leafNodes[k])[0]
            if len(pIdBTN) == 0:
                origin_ii1[k] = 0
                origin_ii0[k] = 0
            else:
                reBTN = xRelatedLabel[pIdBTN]
                # numreBTN =
                origin_ii1[k] = np.sum(np.where(reBTN == 1)[0])
                origin_ii0[k] = np.sum(np.where(reBTN == 0)[0])
            # origin_ii0[k] =len(pIdBTN) - origin_ii1[k]

        tau1_ori = tau1_k - origin_ii1
        tau0_ori = tau0_k - origin_ii0
        if np.sum(np.where(tau0_ori < 0)[0]) > 0:
            print('----------------------Error When Update XX!!!!!!!!!!!!!!!')

        # Here in the matlab version, propoxCoor is calculated from equation
        # Do we need to change?
        propo_x_coor = newXCoordinates[x]
        newLineX = np.repeat(np.array([propo_x_coor]), dataNum)

        newLineCoor = np.array([newLineX, yDataCoor]).T

        nodesBelongTo = tree.dts_coor_belong(newLineCoor, tree.node_num_seq[-1])

        propo_ii1 = np.zeros(numOfClass)

        propo_ii0 = np.zeros(numOfClass)

        for k in range(numOfClass):
            newPIdBTN = np.where(nodesBelongTo == leafNodes[k])[0]
            if len(newPIdBTN) == 0:
                propo_ii1[k] = 0
                propo_ii0[k] = 0
            else:
                newRelationBTN = nodesBelongTo[newPIdBTN]
                propo_ii1[k] = np.sum(np.where(newRelationBTN == 1)[0])
                propo_ii0[k] = np.sum(np.where(newRelationBTN == 0)[0])
            # propo_ii0[k] = len(newPIdBTN) - propo_ii1[k]

        tau1_propo = tau1_ori + propo_ii1
        tau0_propo = tau0_ori + propo_ii0
        if np.sum(np.where(tau0_propo < 0)[0]) > 0:
            print('----------------------Error!!!!!!!!!!!!!!!')
        if np.log(rand[x, 0]) < (np.sum(scipy.special.gammaln(beta + tau1_propo) + scipy.special.gammaln(
                beta + tau0_propo) - scipy.special.gammaln(
                beta + tau1_propo + beta + tau0_propo) - scipy.special.gammaln(beta + tau1_k) - scipy.special.gammaln(
                beta + tau0_k) + scipy.special.gammaln(beta + tau1_k + beta + tau0_k))):
            tau0_k = tau0_propo
            tau1_k = tau1_propo

            xDataCoor[x] = propo_x_coor
            tree.z_label[x, :] = nodesBelongTo
        else:
            # Refuse
            # All the variables related return to the previous status
            pass

    for y in range(dataNum):
        yRelatedLabel = z_label[:, y]
        yorigin_ii1 = np.zeros(numOfClass)
        yorigin_ii0 = np.zeros(numOfClass)
        for k in range(numOfClass):
            ypIdBTN = np.where(yRelatedLabel == leafNodes[k])[0]
            if len(ypIdBTN) == 0:
                yorigin_ii0[k] = 0
                yorigin_ii1[k] = 0
            else:
                yreBTN = yRelatedLabel[ypIdBTN]
                yorigin_ii1[k] = np.sum(np.where(yreBTN == 1)[0])
                yorigin_ii0[k] = np.sum(np.where(yreBTN == 0)[0])
            # yorigin_ii0[k] = len(ypIdBTN)- yorigin_ii1[k]
        ytau1_ori = tau1_k - yorigin_ii1
        ytau0_ori = tau0_k - yorigin_ii0

        propo_y_coor = newYCoordinates[y]
        newLineY = np.repeat(np.array([propo_y_coor]), dataNum)

        ynewLineCoor = np.array([xDataCoor, newLineY]).T

        ynodesBelongTo = tree.dts_coor_belong(ynewLineCoor, tree.node_num_seq[-1])
        ypropo_ii1 = np.zeros(numOfClass)
        ypropo_ii0 = np.zeros(numOfClass)

        for k in range(numOfClass):
            ynewPIdBTN = np.where(ynodesBelongTo == leafNodes[k])[0]
            if len(ynewPIdBTN) == 0:
                ypropo_ii1[k] = 0
                ypropo_ii0[k] = 0
            else:
                ynewRelationBTN = ynodesBelongTo[ynewPIdBTN]
                ypropo_ii1[k] = np.sum(np.where(ynewRelationBTN == 1)[0])
                ypropo_ii0[k] = np.sum(np.where(ynewRelationBTN == 0)[0])
            # ypropo_ii0[k] = len(ynewPIdBTN) - ypropo_ii1[k]
        ytau1_propo = ytau1_ori + ypropo_ii1
        ytau0_propo = ytau0_ori + ypropo_ii0
        if np.sum(np.where(ytau0_propo < 0)[0]) > 0:
            print('---------------Error When Update Y !!!!!!!!!!!!')
        if np.log(rand[y, 1]) < (np.sum(scipy.special.gammaln(beta + ytau1_propo) + scipy.special.gammaln(
                beta + ytau0_propo) - scipy.special.gammaln(
                beta + ytau1_propo + beta + ytau0_propo) - scipy.special.gammaln(beta + tau1_k) - scipy.special.gammaln(
                beta + tau0_k) + scipy.special.gammaln(beta + tau1_k + beta + tau0_k))):
            tau0_k = ytau0_propo
            tau1_k = ytau1_propo
            yDataCoor[y] = propo_y_coor
            tree.z_label[:, y] = ynodesBelongTo
        else:
            pass

    tau_k = tau0_k + tau1_k
    linkProb = (tau1_k + beta) / (tau_k + 2 * beta)
    # Start to calculate AUC
    # print('tau0_k', tau0_k)

    # print('tau_k', tau_k)

    # Start to calculate trainll
    for x in range(numOfClass):
        # pointIdBelongToNode=np.where(prop_z == leafNodes[i])[0]
        # relationBelongToNode = relation[pointIdBelongToNode]
        ii_data = relation[np.where(prop_z == leafNodes[x])[0]]
        trainll += np.sum(np.where(ii_data == 1)[0]) * np.log(linkProb[x]) + np.sum(np.where(ii_data == 0)[0]) * np.log(
            1 - linkProb[x])
    # print('numOfClass', numOfClass)

    return (trainll, auc, xDataCoor, yDataCoor)

#tree_stage_cut
def tree_stage_cut(dts_star, stage_i):
    if stage_i < len(dts_star.candidate_set):
        # By Caoyuan get the first dts_star.node_num_seq[stage_i]+1 rows
        dts_star.treev = dts_star.treev[:(dts_star.node_num_seq[stage_i] + 1), ]
        condition_judge = dts_star.treev[:, 1] > dts_star.treev[-1, 0]
        # By Caoyuan 1,2,3,4  left node, right node, cut dimension, cut value

        dts_star.treev[condition_judge, 1] = 0
        dts_star.treev[condition_judge, 2] = 0
        dts_star.treev[condition_judge, 3] = 0
        dts_star.treev[condition_judge, 4] = 0

        dts_star.treebound_lower = dts_star.treebound_lower[:(dts_star.node_num_seq[stage_i] + 1), :]
        dts_star.treebound_upper = dts_star.treebound_upper[:(dts_star.node_num_seq[stage_i] + 1), :]
        dts_star.candidate_set = copy.deepcopy(dts_star.candidate_set[:(stage_i + 1)])
        dts_star.node_num_seq = copy.deepcopy(dts_star.node_num_seq[:(stage_i + 1)])
        dts_star.remaining_budget_seq = copy.deepcopy(dts_star.remaining_budget_seq[:(stage_i + 1)])
        dts_star.remaining_budget = dts_star.remaining_budget_seq[stage_i]
    return dts_star
