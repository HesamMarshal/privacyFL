from enum import unique
from traceback import print_tb
import config
import datetime
import numpy as np
import pandas as pd
import pickle
import random
from client_agent import ClientAgent
from server_agent import ServerAgent
from directory import Directory
from sklearn.datasets import load_digits
from utils import data_formatting
from utils.print_config import print_config
from utils.model_evaluator import ModelEvaluator
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import os

# https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset/version/1


class Initializer:
    def __init__(self, num_clients, num_servers, iterations):
        """
        Offline stage of simulation. Initializes clients and servers for iteration as well as gives each client its data.
        :param num_clients: number of clients to be use for simulation
        :param num_servers: number of servers to be use for simulation. Personalized coding required if greater than 1.
        :param iterations: number of iterations to run simulation for
        """

        global len_per_iteration

        filepath = ""

        if config.DATASET == "KDDCUP":
            # TODO: Explain
            # KDDCUP 99
            ############################################################
            file_path = os.path.join(os.getcwd(), 'datasets', 'kddcup.csv')

            dataset = pd.read_csv(file_path)
            print(f"Dataset: KDD CUP")
            print(
                f"Dataset Rows: {dataset.shape[0]}\nDataset Cols: {dataset.shape[1]} ")
            print(f"Classifier: {config.CLASSIFIER}")

            x = dataset.drop(dataset.iloc[:, 41:42], axis=1)
            x = x.drop(x.iloc[:, 1:4], axis=1)
            x = np.array(x)
            y = dataset.iloc[:, 41].apply(lambda x: 1 if x == 'normal.' else 0)

            y = np.array(y)
            scaler = MinMaxScaler()
            scaler.fit(x)
            X = scaler.transform(x)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=0)

            # X_train, X_test = X[:-config.LEN_TEST], X[-config.LEN_TEST:]
            # y_train, y_test = y[:-config.LEN_TEST], y[-config.LEN_TEST:]
            #####################################################################
        elif config.DATASET == "LIVING":
            # TODO: Explain
            # Living_conditions_survey_only_2009_15_Cols
            #####################################################################

            file_path = os.path.join(
                os.getcwd(), 'datasets', 'Living_conditions_survey_only_2009_15_Cols_6201.csv')

            dataset = pd.read_csv(file_path)
            print(f"Dataset: Living Conditions Survey")
            print(f"Dataset Rows: {dataset.shape[0]}")
            print(f"Dataset Cols: {dataset.shape[1]}")
            print(f"Classifier: {config.CLASSIFIER}")

            col = 'Sp63b_09'  # Number of sick leave periods > 14 days
            y = dataset[col]
            # print(y.unique())
            # y = dataset.iloc[:, 41].apply(lambda x: 1 if x == 'normal.' else 0)
            y = np.array(y)

            x = dataset.drop(col, axis=1)
            x = np.array(x)

            scaler = MinMaxScaler()
            scaler.fit(x)
            X = scaler.transform(x)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=0)

            # print(f"{pd.unique(y_train)}")
            if X.shape[0] != y.shape[0]:
                print("X and y rows are mismatched, check dataset again")
            #####################################################################
        elif config.DATASET == "MNIST":
            # MNIST
            ############################################################
            digits = load_digits()  # using sklearn's MNIST dataset
            print('----------------------', digits.data.shape)
            X, y = digits.data, digits.target

            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            X_train, X_test = X[:-config.LEN_TEST], X[-config.LEN_TEST:]
            y_train, y_test = y[:-config.LEN_TEST], y[-config.LEN_TEST:]
            ############################################################

        elif config.DATASET == "SPINE":
            # SPINE
            ############################################################
            file_path = os.path.join(
                os.getcwd(), 'datasets', 'Dataset_spine.csv')

            dataset = pd.read_csv(file_path)
            dataset = dataset.drop(dataset.iloc[:, 13:14], axis=1)
            print(f"Dataset: Spine DataSet")
            print(
                f"Dataset Rows: {dataset.shape[0]}\nDataset Cols: {dataset.shape[1]} ")
            print(f"Classifier: {config.CLASSIFIER}")
            x = dataset.drop(['Class_att'], axis=1)
            x = np.array(x)
            if config.VERBOSE_DEBUG:
                print(x[2])
            y = dataset.iloc[:, 12].apply(lambda x: 1 if x == 'Normal' else 0)

            y = np.array(y)
            scaler = MinMaxScaler()
            scaler.fit(x)
            X = scaler.transform(x)

            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=config.TEST_SIZE, random_state=0)

        #################################################################

        # extract only amount that we require
        # number_of_samples = 0
        # for client_name in config.client_names:
        #     len_per_iteration = config.LENS_PER_ITERATION[client_name]
        #     number_of_samples += len_per_iteration * iterations

        # X_train = X_train[:number_of_samples]
        # y_train = y_train[:number_of_samples]

        # client_to_datasets = data_formatting.partition_data(X_train, y_train, config.client_names, iterations,
        #                                                     config.LENS_PER_ITERATION, cumulative=config.USING_CUMULATIVE)

        # extract only amount that we require
        number_of_samples = 0
        # print(f"{config.LENS_PER_ITERATION=}")
        for client_name in config.client_names:
            len_per_iteration = config.LENS_PER_ITERATION[client_name]
            number_of_samples += len_per_iteration * iterations
            if config.VERBOSE_DEBUG:
                print(f"------------- {client_name} : {number_of_samples=}")
        if config.VERBOSE_DEBUG:
            print(f"------------- {len(X_train)=}")
            print(f"------------- {len(y_train)=}")

        ratio = number_of_samples / len(X_train)
        # print(ratio)
        # TODO :
        X_train = X_train[:number_of_samples]
        y_train = y_train[:number_of_samples]

        if config.VERBOSE_DEBUG:
            print(f"{len(X_train)=}")
            print(f"{len(y_train)=}")
            print(f"{pd.unique(y_train)=}")
            print(f"{pd.unique(y_test)=}")

        client_to_datasets = data_formatting.partition_data(X_train, y_train,
                                                            config.client_names,
                                                            iterations,
                                                            config.LENS_PER_ITERATION,
                                                            cumulative=config.USING_CUMULATIVE)

        # print_config(len_per_iteration=
        # config.LEN_PER_ITERATION)
        # print(f"{type(client_to_datasets)=}")
        # print(f"{client_to_datasets=}")

        print('\n \n \nSTARTING SIMULATION \n \n \n')

        active_clients = {'client_agent' + str(i) for i in range(num_clients)}
        if config.VERBOSE_DEBUG:
            print(f"{active_clients=}")

        self.clients = {
            'client_agent' + str(i): ClientAgent(agent_number=i,
                                                 train_datasets=client_to_datasets['client_agent' + str(
                                                     i)],
                                                 evaluator=ModelEvaluator(
                                                     X_test, y_test),
                                                 active_clients=active_clients) for i in
            range(num_clients)}  # initialize the agents

        self.server_agents = {'server_agent' + str(i): ServerAgent(agent_number=i) for i in
                              range(num_servers)}  # initialize servers

        # create directory with mappings from names to instances
        self.directory = Directory(
            clients=self.clients, server_agents=self.server_agents)

        for agent_name, agent in self.clients.items():
            agent.set_directory(self.directory)
            agent.initializations()
        for agent_name, agent in self.server_agents.items():
            agent.set_directory(self.directory)

        # OFFLINE diffie-helman key exchange
        # NOTE: this is sequential in implementation, but simulated as occuring parallel
        if config.USE_SECURITY:
            # measuring how long the python script takes
            key_exchange_start = datetime.datetime.now()
            max_latencies = []
            for client_name, client in self.clients.items():
                # not including logic of sending/receiving public keys in latency computation since it is nearly zero
                client.send_pubkeys()
                max_latency = max(config.LATENCY_DICT[client_name].values())
                max_latencies.append(max_latency)
            simulated_time = max(max_latencies)

            key_exchange_end = datetime.datetime.now()  # measuring runtime
            key_exchange_duration = key_exchange_end - key_exchange_start
            simulated_time += key_exchange_duration
            if config.SIMULATE_LATENCIES:
                print(
                    'Diffie-helman key exchange simulated duration: {}\nDiffie-helman key exchange real run-time: {}\n'.format(
                        simulated_time, key_exchange_duration))

            for client_name, client in self.clients.items():
                client.initialize_common_keys()

    def run_simulation(self, num_iterations, server_agent_name='server_agent0'):
        """
        Online stage of simulation.
        :param num_iterations: number of iterations to run
        :param server_agent_name: which server to use. Defaults to first server.
        """
        # ONLINE
        server_agent = self.directory.server_agents[server_agent_name]
        server_agent.request_values(num_iterations=num_iterations)
        server_agent.final_statistics()
