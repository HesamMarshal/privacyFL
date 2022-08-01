from distutils.command.config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config


file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'fed-I10-E0.01.csv')
# file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'fed-I10-E0.001.csv')
fed_1 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'fed-I10-E0.1.csv')
fed_2 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'fed-I10-E1.0.csv')
fed_3 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'fed-I10-E8.0.csv')
fed_4 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'client-I10-E0.01.csv')
client_1 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'client-I10-E0.1.csv')
client_2 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'client-I10-E1.0.csv')
client_3 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'client-I10-E8.0.csv')
# file_path = os.path.join(os.getcwd(), 'src', 'fig3', 'client-I10-E1000.0.csv')

client_4 = pd.read_csv(file_path)


mean_client = []
for i in range(len(client_1)):
    mean_client.append(np.mean(
        [client_1.iloc[i][1], client_2.iloc[i][1], client_3.iloc[i][1], client_4.iloc[i][1]]))

print(mean_client)
plt.figure(figsize=(10, 10))
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mean_client,
         'black', label='Mean Client Accuracy',)
plt.legend()
# print(fed_1['itraton'])
# plt.plot(fed_1['itration'], fed_1['accuracy'], 'b--',
#          label='Federated Accuracy Epsilon = 0.01')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fed_1['accuracy'], 'b--',
         label='Federated Accuracy Epsilon = 0.01')

plt.legend()

# plt.plot(fed_2['itration'], fed_2['accuracy'], 'g--',
#          label='Federated Accuracy Epsilon = 0.1')

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fed_2['accuracy'], 'g--',
         label='Federated Accuracy Epsilon = 0.1')
plt.legend()

# plt.plot(fed_3['itration'], fed_3['accuracy'], 'r--',
#          label='Federated Accuracy Epsilon = 1.0')

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fed_3['accuracy'], 'r--',
         label='Federated Accuracy Epsilon = 1.0')
plt.legend()

# plt.plot(fed_4['itration'], fed_4['accuracy'], 'm--',
#          label='Federated Accuracy Epsilon = 8.0')

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fed_4['accuracy'], 'm--',
         label='Federated Accuracy Epsilon = 8.0')
plt.legend()

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title(f"Fig3-{config.CLASSIFIER}-{config.DATASET}")
# plt.ylim(0.4,1)
# plt.show()
plt.savefig(f'fig3-{config.CLASSIFIER}-{config.DATASET}.png')
