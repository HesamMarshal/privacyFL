import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config


file_path = os.path.join(os.getcwd(), 'src', 'fig4', 'acc-N10-E0.001.csv')
acc_1 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig4', 'acc-N50-E0.01.csv')
acc_2 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig4', 'acc-N100-E0.01.csv')
acc_3 = pd.read_csv(file_path)

file_path = os.path.join(os.getcwd(), 'src', 'fig4', 'acc-N100-E1.0.csv')
acc_4 = pd.read_csv(file_path)

# file_path = os.path.join(os.getcwd(), 'src', 'fig4', 'acc-N10-E8.csv')
# acc_5 = pd.read_csv(file_path)


# mean_client = []
# for i in range(len(client_1)):
#     mean_client.append(np.mean(
#         [client_1.iloc[i][1], client_2.iloc[i][1], client_3.iloc[i][1], client_4.iloc[i][1]]))

# print(mean_client)
plt.figure(figsize=(10, 10))

# plt.plot(mean_client, 'black', label='Mean Client Accuracy')
# plt.legend()

# plt.plot(acc_1['itration'], acc_1['accuracy'], 'b--',
#          label='Eps=0.001, N=10')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], acc_1['accuracy'], 'b--',
         label='Eps=0.001, N=10')
plt.legend()

# plt.plot(acc_2['itration'], acc_2['accuracy'], 'g--',
#          label='Eps=0.01, N=50')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], acc_2['accuracy'], 'g--',
         label='Eps=0.01, N=50')
plt.legend()

# plt.plot(acc_3['itration'], acc_3['accuracy'], 'r--',
#          label='Eps=0.01, N=100')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], acc_3['accuracy'], 'r--',
         label='Eps=0.01, N=100')
plt.legend()

# plt.plot(acc_4['itration'], acc_4['accuracy'], 'm--',
#          label='Eps=1.0, N=100')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], acc_4['accuracy'], 'm--',
         label='Eps=1.0, N=100')
plt.legend()

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title(f"Fig4-Classifier: {config.CLASSIFIER}- Dataset: {config.DATASET}")
# plt.ylim(0.4,1)
# plt.show()
plt.savefig(f'fig4-{config.CLASSIFIER}-{config.DATASET}.png')
# plt.savefig('fig3.png')
