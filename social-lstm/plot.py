import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pprint

plot = False

folder_path = "visualize_it"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

obj = pickle.load(open("./plot/SOCIALLSTM/LSTM/test/biwi_eth.pkl", "rb"))
with open("out.txt", "a") as f:
    pprint.pprint(obj, stream=f)
df_o = pd.read_csv("./out.txt", sep=" ")
print(df_o)


# data paths to each dataset result folder
data_path_list = ["./result/SOCIALLSTM/LSTM/biwi/biwi_eth.txt", "./result/SOCIALLSTM/LSTM/crowds/crowds_zara01.txt", "./result/SOCIALLSTM/LSTM/stanford/coupa_0.txt", "./result/SOCIALLSTM/LSTM/stanford/coupa_1.txt", "./result/SOCIALLSTM/LSTM/stanford/gates_2.txt", "./result/SOCIALLSTM/LSTM/stanford/hyang_0.txt", "./result/SOCIALLSTM/LSTM/stanford/hyang_1.txt", "./result/SOCIALLSTM/LSTM/stanford/hyang_3.txt", "./result/SOCIALLSTM/LSTM/stanford/hyang_8.txt", "./result/SOCIALLSTM/LSTM/stanford/little_0.txt", "./result/SOCIALLSTM/LSTM/stanford/little_1.txt", "./result/SOCIALLSTM/LSTM/stanford/little_2.txt", "./result/SOCIALLSTM/LSTM/stanford/little_3.txt", "./result/SOCIALLSTM/LSTM/stanford/nexus_0.txt", "./result/SOCIALLSTM/LSTM/stanford/nexus_6.txt", "./result/SOCIALLSTM/LSTM/stanford/quad_0.txt", "./result/SOCIALLSTM/LSTM/stanford/quad_1.txt", "./result/SOCIALLSTM/LSTM/stanford/quad_2.txt", "./result/SOCIALLSTM/LSTM/stanford/quad_3.txt",]

if plot:
    for data_path in data_path_list:
        # create dataframe of result
        df = pd.read_csv(data_path, sep=" ", names=['frame', 'id', 'x1', 'x2'])

        start_idx = 0  # start index for rows of dataframe
        steps = 20     # number of trajectory points for each agent

        total_ids = len(df.id.value_counts()) # total number of ids(agent)

        for i in range(total_ids):
            x1 = list(df.iloc[start_idx:start_idx+steps, :]['x1'])
            x2 = list(df.iloc[start_idx:start_idx+steps, :]['x2'])
            
            id = df.iloc[start_idx]['id']
            tag = str(data_path[data_path.rfind('/')+1:data_path.rfind('.')] + ' id=' + str(id))    
            
            plt.figure()
            plt.scatter(x1[0:8], x2[0:8], color='orange', label="Observations")
            plt.scatter(x1[8:20], x2[8:20], color='b', label="Predictions")
            plt.legend()
            # plt.xlim(-8, 18)
            # plt.ylim(-11, 15)
            plt.title(tag)
            plt.xlabel("x1 (m)")
            plt.ylabel("x2 (m)")
            plt.savefig("{}/traj_{}.png".format(folder_path, tag))
            plt.close()
            
            start_idx = start_idx+steps