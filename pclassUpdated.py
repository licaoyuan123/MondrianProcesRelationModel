import numpy as np
import copy
import scipy.io
from utility import *


class Tree(object):

    def __init__(self, beta, max_stage, budget, dim_num, data_num, likelihood_ratio = 1):
        super(Tree, self).__init__()
        self.treev = np.zeros([1, 7])
        # 0 node num, 1 left 2 right, 3 cut dimension, 4 cut value 5 cut level 6 perimeter

        self.dim_num = dim_num
        self.beta = beta
        self.max_stage = max_stage
        self.remaining_budget =budget
        self.remaining_budget_seq =[budget]
        self.llRatio_seq = [likelihood_ratio]

        self.data_num = data_num
        self.treev[0, 6] = 2*dim_num
        self.treebound_lower = np.array([np.array(np.zeros(dim_num))])
        self.treebound_upper = np.array([np.array(np.ones(dim_num))])
        self.node_num_seq = [0]
        self.candidate_set = [[0]]

    def visualize(self):
        print('Number of leaf nodes: ', len(self.candidate_set[-1]))
        print('Last leaf node: ', self.node_num_seq)
        print('all level of leaf nodes: ', self.candidate_set)
        # print('All Lower bound: ', self.treebound_lower)
        # print('All upper bound: ', self.treebound_upper)

    def propose_cut(self):
        cu_canset = copy.deepcopy(self.candidate_set[-1])
        perimeter_of_every_node = np.sum(self.treebound_upper[cu_canset] - self.treebound_lower[cu_canset], axis=1)
        perimeter_sum = np.sum(perimeter_of_every_node)
        #correct according to debug by Caoyuan
        cost = -np.log(1 - np.random.uniform()) / perimeter_sum
        remaining_budget = self.remaining_budget- cost

        if remaining_budget < 0:
            # budget is not enough to cut anymore.
            self.remaining_budget = -1
            return -1
        else:
            # propose cut
            self.remaining_budget = remaining_budget
            spe_node_num = copy.deepcopy(self.node_num_seq[-1])
            prob_of_node = perimeter_of_every_node / perimeter_sum
            intra_node = np.random.choice(cu_canset, p=prob_of_node)

            lengthOfEachDimNorm = np.array([self.treebound_upper[intra_node][0] - self.treebound_lower[intra_node][0], self.treebound_upper[intra_node][1] - self.treebound_lower[intra_node][1]])
            totalLength = np.sum(lengthOfEachDimNorm)
            cutPoint  = np.random.uniform(0, totalLength)
            if cutPoint < lengthOfEachDimNorm[0]:
                #3 cut dimension, 4 cut value
                self.treev[intra_node, 3] = 0
                self.treev[intra_node, 4] = self.treebound_lower[intra_node][0] + cutPoint
            else:
                self.treev[intra_node, 3] = 1
                # position[1] - total-cutlength
                self.treev[intra_node, 4] = self.treebound_upper[intra_node][1] - (totalLength - cutPoint)

            add_treev = np.zeros([2, 7])
            add_treev[0, 0] = spe_node_num + 1
            add_treev[1, 0] = spe_node_num + 2
            #By Caoyuan 5 is the cut level

            add_treev[0, 5] = self.treev[intra_node, 5] + 1
            add_treev[1, 5] = self.treev[intra_node, 5] + 1
            self.treev[intra_node, [1, 2]] = np.array([1, 2]) + spe_node_num
            self.treev = np.vstack((self.treev, add_treev))
            added_treebound_lower = np.tile(self.treebound_lower[intra_node], (2, 1))
            added_treebound_upper = np.tile(self.treebound_upper[intra_node], (2, 1))
            #By Caoyuan 3 is cut dimension
            added_treebound_upper[0, self.treev[intra_node, 3].astype(int)] = self.treev[intra_node, 4]
            added_treebound_lower[1, self.treev[intra_node, 3].astype(int)] = self.treev[intra_node, 4]
            self.treebound_lower = np.vstack((self.treebound_lower, added_treebound_lower))
            self.treebound_upper = np.vstack((self.treebound_upper, added_treebound_upper))
            cu_canset.extend([1 + spe_node_num, 2 + spe_node_num])
            spe_node_num += 2
            cu_canset.remove(intra_node)
            self.node_num_seq.append(spe_node_num)
            self.candidate_set.append(cu_canset)
            self.remaining_budget_seq.append(remaining_budget)
            return 0

    def tree_full_gen(self):
        # After treefull gen, there is no need to store the original budget
        # originalBudget = self.budget
        for _ in np.arange(self.max_stage):
            result = self.propose_cut()
            if result == -1:
                break

    def ll_cal(self, data_coor, relation, stage_i):
        # we need to define the likelihood here for cutting the block
        # By Caoyuan calculate the log likelihood of stage_i
        # mi_dts_val_data = self.assign_to_data(xdata, stage_i)
        terminal_node = self.dts_coor_belong(data_coor, stage_i)
        beta = self.beta
        # Returns each data belongs to which node then calculate the likelihood
        # predicted_yval = add_dts_other_val + mi_dts_val_data

        # Use terminal_node and relation to calculate the loglikelihood
        # But not sure about the equation

        # Use the classical calculate loglikelihood equation
        kCluster, ind = np.unique(terminal_node, return_inverse=True)
        lenKCluster = len(kCluster)
        m1 = np.zeros(lenKCluster)
        m0 = np.zeros(lenKCluster)
        mSum = np.zeros(lenKCluster)
        # prob = np.zeros(lenKCluster)

        # Can this for loop be optimized?
        # ll=0
        for k in range(lenKCluster):
            m = np.where(np.array(terminal_node) == kCluster[k])[0]
            # m = np.where(terminal_node ==kCluster[k])
            # print("kCluster:  ----", k, "totalValue", m)
            # mSum[k] = len(m[0])
            mSum[k] = len(m)
            m0[k] = len(np.where(relation[m] == 0)[0])
            m1[k] = len(np.where(relation[m] == 1)[0])

            #new added code:
            # prob[k] = (m1[k]+beta) / (mSum[k] + 2*beta)
            # ll += m1[k] * np.log(prob[k]) + m0[k] * np.log(1-prob[k])
        # m0 = mSum - m1
        mtest = mSum-m0-m1
        if np.sum(mtest) != 0:
            print('-------------ERROR--------------')
            print('-------------ERROR--------------')
            return 'error stop here'
#original correct code
        ll = lenKCluster * (scipy.special.gammaln(2 * beta) - 2 * scipy.special.gammaln(beta)) + np.sum(
            scipy.special.gammaln(beta + m1) + scipy.special.gammaln(beta + m0) - scipy.special.gammaln(
                2 * beta + mSum))
        # ll = (scipy.special.gammaln(2 * beta) - 2 * scipy.special.gammaln(beta)) + np.sum(
        #     scipy.special.gammaln(beta + m1) + scipy.special.gammaln(beta + m0) - scipy.special.gammaln(
        #         2 * beta + mSum))


        # Here is correct, m0 is a vector , sum to small less than relation num
        # print('m0: ', m0)
        # raw_input()

        # for x in xrange(len(terminal_node)):
        #     for k in xrange(len(kCluster)):
        #         #if relation[ind[x]]==1 and :
        #         if terminal_node[x]==kCluster[k]:
        #             if relation[x] ==1:
        #                 m1[k] += 1
        #             else:
        #                 m0[k] += 1

        # ll = 0
        # ll = np.sum(scipy.special.gammaln(beta+m1)) + np.sum(scipy.special.gammaln(beta+m0))- np.sum(scipy.special.gammaln(2* beta+mSum))+ lenKCluster *(scipy.special.gammaln(2* beta) - 2 * scipy.special.gammaln(beta) )
        # print('Before loglikelihood is: ', ll)
        # ll = 0
        # for k in xrange(lenKCluster):
        # print('scipy.special.beta(m1[k]+beta, m0[k]+beta): ', scipy.special.beta(m1[k]+beta, m0[k]+beta))
        # ll += np.log(scipy.special.beta(m1[k]+beta, m0[k]+beta) ) - np.log(scipy.special.beta(beta, beta))
        # for k in xrange(len(kCluster)):
        #     ll = ll+ scipy.special.gammaln(beta+ m1[k]) + scipy.special.gammaln(beta+ m0[k]) - scipy.special.gammaln(2* beta + m1[k]+m0[k])
        # print('After loglikelihood is: ', ll)
        # raw_input()
        return ll

    def dts_coor_belong(self, coors, stage_i):
        # By Caoyuan Calculate each data belog to which leaf node
        # By Caoyuan: no.2 is r node index
        # stage_i is the  number of leaf nodes of last step
        aa = copy.deepcopy(self.treev[:(stage_i + 1), 2])
        # By Caoyuan set all the element of aa> stage_i to 0
        aa[aa > stage_i] = 0
        # IF the right child node number is 0, then the current node is a leaf node
        # print("======================aa", aa)
        # print("======================np.where(aa==0)", np.where(aa==0))
        terminal_index = np.where(aa == 0)[0]
        # Explain: np.where returns 2 dimension result, the first dimension is array, the second is non
        # print("======================terminal_index", terminal_index)
        # By Caoyuan is this the first node to be cutted for the next level?
        terminal_lower = self.treebound_lower[terminal_index]
        # print("======================terminal_lower", terminal_lower)
        terminal_upper = self.treebound_upper[terminal_index]

        if len(coors.shape) == 1:
            # By Caoyuan, after reshapping, become a two dimension matrix
            indicator_index = (coors.reshape((1, -1)) >= terminal_lower) & ((coors.reshape((1, -1)) <= terminal_upper))
            # print("======================indicator_index", indicator_index)
            # By Caoyuan, return a vector of true or false
            nonempty_index = np.dot(indicator_index.T, np.arange(len(terminal_lower)))
            # print("======================nonempty_index", nonempty_index)
        else:
            compare_mat_lower = ((coors[np.newaxis].swapaxes(1, 0) - terminal_lower[np.newaxis]) >= 0)
            compare_mat_upper = ((coors[np.newaxis].swapaxes(1, 0) - terminal_upper[np.newaxis]) < 0)
            indicator_index = (np.prod(compare_mat_lower, axis=2)) * (np.prod(compare_mat_upper, axis=2))
            # By Caoyuan Calculate each data belog to which leaf node

            nonempty_index = np.dot(indicator_index, np.arange(len(terminal_lower)))
            #
        terminal_node = terminal_index[nonempty_index]
        # print('===================len(terminal_node)', len(terminal_node))
        # By Caoyuan Result is 200, so here terminal_node means each data belongs to which node
        # print('Each data belong to : leaf nodes', terminal_node )
        # raw_input()

        return terminal_node

    def update_particle(self, particle_num, current_tree, max_stage, budget, data_coor, relation, dim_num):
        # By Caoyuan Here can use self instead of dts_star to save memory
        # print("In dts_update Now==")
        par_dts_seq = []
        # By Caoyuan Generate Another particleNum trees
        # def __init__(self, mTree,maxStage, beta, budget, dimNum, data):
        for pari in range(particle_num):
            # Change from Here
            # (self, beta, max_stage, budget, dim_num, data_num, likelihood_ratio = 1):
            # dts_pari = Tree(self.mTree, max_stage, self.beta, budget, dimNum)
            dts_pari = Tree(self.beta, max_stage, budget, dim_num, self.data_num, likelihood_ratio = 1)
            par_dts_seq.append(dts_pari)

        # by Caoyuan now, the number of trees is particleNUm + 1
        previous_ll = np.zeros(particle_num + 1)
        hd_i = 0
        continue_flag = True
        # Here ignore the max stage first for similarity
        # for hd_i in np.arange(1, maxstage_PG, 1):
        # while(continue_flag and hd_i<maxStage ):
        while continue_flag:
            #print('--In while--')
            hd_i += 1
            continue_flag = False
            ll_seqi = []
            for pari in range(particle_num):
                # Every Tree cuts one level down
                # print(par_dts_seq[pari].candidate_set)

                # if (hd_i >= len(par_dts_seq[pari].candidate_set)) & (par_dts_seq[pari].candidate_set[-1]==[]):
                # By Caoyuan par_dts_seq stores new generated trees number is particleNum
                # print("==IN UPDATE ==par_dts_seq[pari].candidate_set[-1]: ======", par_dts_seq[pari].candidate_set[-1])

                if par_dts_seq[pari].remaining_budget > 0:
                    # print('remaining budeget larger than 0 once')
                    continue_flag = True
                    # propose_cut(self, beta, budget, dimNum, data):
                    # print('In Update and proposed cut:********')
                    # print('In dts_update, Before propose_cut, budget', par_dts_seq[pari].budget)
                    par_dts_seq[pari].propose_cut()
                    # print('In dts_update, After propose_cut, budget', par_dts_seq[pari].budget)
                    # Not stucked here
                    # print("====Leaf nodes After cut:", par_dts_seq[pari].candidate_set[-1] )

                ll_val_i = par_dts_seq[pari].ll_cal(data_coor, relation, par_dts_seq[pari].node_num_seq[-1])
                ll_seqi.append(ll_val_i)
            # likelihood is different

            if hd_i >= len(self.candidate_set):
                ll_val_i = (self.ll_cal(data_coor, relation, self.node_num_seq[-1]))
            else:
                continue_flag = True
                ll_val_i = (self.ll_cal(data_coor, relation, self.node_num_seq[hd_i]))

            ll_seqi.append(ll_val_i)
            # print('Likelihood of all trees: ll_seqi--->', ll_seqi)

            # We might not use previous_ll
            # ll_seqi_ratio = ll_seqi-previous_ll
            # By Caoyuan   divide the previous step probability
            ll_seqi_ratio = ll_seqi - previous_ll
            # ll_seqi_ratio = ll_seqi
            # print('Likelihood of all trees: ll_seqi--->', ll_seqi)

            # normalize the log-likelihood
            propb = np.exp(ll_seqi_ratio - np.max(ll_seqi_ratio))
            #print('----probability is: ', propb/np.sum(propb))

            # print('probability to chose Nodes: ', propb)
            # raw_input()

            copy_dts_star = copy.deepcopy(current_tree)
            # By Caoyuan tree_stage_cut, cut the tree to the hd_i th stage
            par_dts_seq.append(tree_stage_cut(copy_dts_star, hd_i))

            if continue_flag:
                # By Caoyuan multinomial?
                # replace=True means it's possible to choice the same tree
                # print("Parameters of multinomial: ........", propb/ np.sum(propb))
                select_index = np.random.choice((particle_num + 1), particle_num, replace=True, p=propb / np.sum(propb))

                par_dts_seq[:particle_num] = [copy.deepcopy(par_dts_seq[i]) for i in select_index]
                previous_ll[:particle_num] = [copy.deepcopy(ll_seqi[i]) for i in select_index]
                previous_ll[-1] = copy.deepcopy(ll_seqi[-1])
                # By Caoyuan, before length of par_dts_seq is 11, previous, update the first
                # particleN of par_dts_seq, the last one is useless
                par_dts_seq.pop()

            else:
                # print("multinomial probability: ", propb / np.sum(propb))
                select_index = np.random.choice((particle_num + 1), replace=True, p=propb / np.sum(propb))
                # print("Parameters of multinomial: ........", propb/ np.sum(propb))
                # print("selected Tree Number: ===", select_index)
                final_particle = par_dts_seq[select_index]
                break
            # Print the tree structure of each particle tree
            # for x in range(particleNUm):
            #     print("par_dts_seq", par_dts_seq[x].candidate_set)
            #     pass
            # print("=====")
        # print('---Out of while---')
        # print('After this update, the number of the nodes in final particle: ', len(self.node_num_seq))
        #self = copy.deepcopy(final_particle)


        self.treev = final_particle.treev
        self.treebound_lower = final_particle.treebound_lower
        self.treebound_upper = final_particle.treebound_upper
        self.node_num_seq = copy.deepcopy(final_particle.node_num_seq)
        self.candidate_set = copy.deepcopy(final_particle.candidate_set)
        self.remaining_budget_seq = copy.deepcopy(final_particle.remaining_budget_seq)
        self.remaining_budget = final_particle.remaining_budget


        #print("====self.candidate_set", self.candidate_set )
        #nodes = self.dts_coor_belong(data, self.node_num_seq[-1])
        # print("=====nodes: ",nodes )

        # Blow calculation could be ignored, let's see the result
        loglikelihood = final_particle.ll_cal(data_coor, relation, final_particle.node_num_seq[-1])
        # loglikelihoodSelf = self.ll_cal(data_coor, relation, self.node_num_seq[-1])
        # if loglikelihood != loglikelihoodSelf:
        #     print('----------------------WRONG!!!!!!!!!!!!')

        return loglikelihood

