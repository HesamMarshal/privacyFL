from traceback import print_tb
import numpy as np
import pandas as pd
import config
import math


def partition_data_2(X, y, client_names, num_iterations, lens_per_iteration, cumulative=False):
    """
    Function used to partition data to give to clients for simulation.
    :type X: numpy array
    :type y: numpy array
    :param client_names: list of all client agents' name
    :param num_iterations: number of iterations to partition data for (initially set in config.py)
    :type num_iterations: int
    :param lens_per_iteration: length of new dataset available each iteration for each client
    :type lens_per_iteration: dictionary
    :param cumulative: flag that indidicates where dataset creation should be cumulative. If True,
    the dataset at iteration i+1 contains all of the data from previous iterations as well.
    :type cumulative: bool
    :return: dictionary mapping each client by name to another dictionary which contains its dataset for each
    iteration mapped by ints (iteration).
    # re code to classify both 0 and 1 in the dataset
    """

    # Devide X and to X1,X0
    # X1 contains Xs with y =1
    # X0 contains Xs with y=0
    X1 = X[y == 1]
    X0 = X[y == 0]

    x1_x0_ratio = math.floor(len(X1)/len(X0)*100)

    if config.VERBOSE_DEBUG:
        print('Len X1 = {} Len X0 = {}'.format(len(X1), len(X0)))
        print('floor of X1/X0 = {}%'.format(x1_x0_ratio))
        print('X1.shape={}  X0.shape= {}'.format(X1.shape,  X0.shape))
        # print(y[0])

    client_datasets = {client_name: None for client_name in client_names}

    # partition each client its data
    last_index = 0  # where to start the next client's dataset from
    number_of_clients = len(client_names)
    if config.VERBOSE_DEBUG:
        print('Number of Clients {}'.format(number_of_clients))

    for client_name in client_names:
        if config.VERBOSE_DEBUG:
            print(f"============================ {client_name=}")
        datasets_i = {}  # datasets for client i
        len_per_iteration = lens_per_iteration[client_name]
        start_idx = last_index
        # where this client's datasets will end
        last_index += num_iterations * len_per_iteration
        if config.VERBOSE_DEBUG:
            print(f"{start_idx=} -------- {last_index=}")

        for j in range(1, num_iterations+1):
            if cumulative:  # dataset gets bigger each iteraton
                end_indx = start_idx + len_per_iteration * j
            else:
                end_indx = start_idx + len_per_iteration  # add the length per iteration

            range_X_ij = end_indx - start_idx

            # end_X0 = end_indx - math.floor(end_indx * x1_x0_ratio/100)
            start_X1 = math.floor(start_idx * x1_x0_ratio/100)
            end_X1 = start_X1 + math.floor(x1_x0_ratio * range_X_ij / 100)
            number_of_1 = end_X1 - start_X1
            number_of_0 = range_X_ij - number_of_1

            y_ij = []  # y[start_idx:end_indx]
            # X1_ij = X1[start_idx: end_X1]
            X1_ij = X1[start_X1: end_X1]
            X0_ij = X0[end_X1+1: end_indx]

            for i in range(number_of_1):
                y_ij.append(1)
            for i in range(number_of_1):
                # y_ij[i+end_indx-1] = 0
                y_ij.append(0)

            # y1_ij[start_idx: end_X1] = 0
            # y0_ij[end_X1+1: end_indx] = 1

            # X_ij = X[start_idx:end_indx]
            # y_ij = y[start_idx:end_indx]

            # print("lenX_ij", len(X_ij))
            # for i in range(len(X_ij)):
            #     print(X_ij[i])
            # print(f"{pd.unique(y_ij)=}")
            X_ij2 = np.concatenate((X0_ij, X1_ij), axis=0)
            y_ij2 = np.array(y_ij)

            # datasets_i[j] = (X_ij, y_ij)
            datasets_i[j] = (X_ij2, y_ij2)

            if config.VERBOSE_DEBUG:
                print('start_X1 {}'.format(start_X1))
                print('end_X1 {}'.format(end_X1))
                print('number_of_1 {}'.format(number_of_1))
                print('number_of_0 {}'.format(number_of_0))
                print('From {} to {}'.format(start_idx, end_indx))
                print('len X_ij= {} '.format(range_X_ij))
                print('X1 From {} to {}'.format(start_idx, end_X1))
                print('X0 From {} to {}'.format(end_X1+1, end_indx))
                print('X1_ij.shape={}  X0_ij.shape= {}'.format(
                    X1_ij.shape,  X0_ij.shape))
                print('X_ij2.shape={} y_ij2.shape= {}'.format(
                    X_ij2.shape,  y_ij2.shape))
                print()

            if not cumulative:
                start_idx = end_indx  # move up start index

        client_datasets[client_name] = datasets_i
        # print(
        # f"============================ {client_datasets['client_agent0']=}")

    return client_datasets


def partition_data(X, y, client_names, num_iterations, lens_per_iteration, cumulative=False):
    """
    Function used to partition data to give to clients for simulation.
    :type X: numpy array
    :type y: numpy array
    :param client_names: list of all client agents' name
    :param num_iterations: number of iterations to partition data for (initially set in config.py)
    :type num_iterations: int
    :param lens_per_iteration: length of new dataset available each iteration for each client
    :type lens_per_iteration: dictionary
    :param cumulative: flag that indidicates where dataset creation should be cumulative. If True,
    the dataset at iteration i+1 contains all of the data from previous iterations as well.
    :type cumulative: bool
    :return: dictionary mapping each client by name to another dictionary which contains its dataset for each
    iteration mapped by ints (iteration).
    """
    client_datasets = {client_name: None for client_name in client_names}
    # partition each client its data
    last_index = 0  # where to start the next client's dataset from
    for client_name in client_names:
        datasets_i = {}  # datasets for client i
        len_per_iteration = lens_per_iteration[client_name]
        start_idx = last_index
        # where this client's datasets will end
        last_index += num_iterations * len_per_iteration
        for j in range(1, num_iterations+1):
            if cumulative:  # dataset gets bigger each iteraton
                end_indx = start_idx + len_per_iteration * j
            else:
                end_indx = start_idx + len_per_iteration  # add the length per iteration
            print('From {} to {}'.format(start_idx, end_indx))
            X_ij = X[start_idx:end_indx]
            y_ij = y[start_idx:end_indx]
            datasets_i[j] = (X_ij, y_ij)
            if not cumulative:
                start_idx = end_indx  # move up start index
        client_datasets[client_name] = datasets_i
    return client_datasets
