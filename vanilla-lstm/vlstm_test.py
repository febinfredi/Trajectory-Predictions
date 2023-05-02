"""
Script to perform model training
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import os
import dataloader 
import model
import utils
import matplotlib.pyplot as plt


PLOT_GRAPHS = True
NUM_OF_PREDICTION_TEST = 20

if __name__ == "__main__":

    # defining model save location
    save_location = "./models"
    # defining dataset locations
    dataset_folder = "./datasets"
    dataset_name = "raw"
    # setting validation size. if val_size = 0, split percentage is 80-20
    val_size = 0
    # length of sequence given to encoder
    gt = 8
    # length of sequence given to decoder
    horizon = 12

    # defining batch size
    batch_size = 64

    # creating torch datasets
    train_dataset, _ = dataloader.create_dataset(dataset_folder, dataset_name, val_size, gt, horizon, delim="\t", train=True)
    test_dataset, _ = dataloader.create_dataset(dataset_folder, dataset_name, val_size, gt, horizon, delim="\t", train=False, eval=True)

    # creating torch dataloaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    
    # calculating the mean and standard deviation of velocities of the entire dataset
    mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
    std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))

    # loading saved model file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_file = torch.load(os.path.join(save_location, 'epoch12.pth'), map_location=torch.device(device))

    # creating model and loading weights
    encoder_ip_size = 2
    decoder_ip_size = 3
    model_op_size = 3
    emb_size = 512
    num_heads = 8
    ff_hidden_size = 2048
    n = 6
    dropout=0.1

    model_loaded = model.TFModel(encoder_ip_size, decoder_ip_size, model_op_size, emb_size, \
                    num_heads, ff_hidden_size, n, dropout=0.1)
    model_loaded = model_loaded.to(device)
    model_loaded.load_state_dict(loaded_file['model_state_dict'])

    # Running the testing loop to generate prediction trajectories on testing data
    testing_loss = []
    test_mad = []
    test_fad = []

    with torch.no_grad():
        # EVALUATION MODE
        model_loaded.eval()
        
        # validation variables
        batch_test_loss=0
        gt = []
        pr = []
        obs = []

        for id_b, data in enumerate(test_loader):  # dtype(id_b) = int dtype(data) = dict (dict_keys(['src', 'trg', 'frames', 'seq_start', 'dataset', 'peds']))
            print(id_b) 
            
            # storing groung truth 
            gt.append(data['trg'][:, :, 0:2])
            obs.append(data['src'][:,:, 0:2])
            # input to encoder input
            test_input = (data['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)

            # input to decoder
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(test_input.shape[0], 1, 1).to(device)
            dec_inp = start_of_seq
            # decoder masks
            dec_source_mask = torch.ones((test_input.shape[0], 1, test_input.shape[1])).to(device)
            dec_target_mask = utils.subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)

            # prediction till horizon length
            for i in range(horizon):
                # getting model prediction
                model_output = model_loaded.forward(test_input, dec_inp, dec_source_mask, dec_target_mask)
                # appending the predicition to decoder input for next cycle
                dec_inp = torch.cat((dec_inp, model_output[:, -1:, :]), 1)

            # calculating loss using pairwise distance of all predictions
            test_loss = F.pairwise_distance(dec_inp[:,1:,0:2].contiguous().view(-1, 2),
                                    ((data['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).\
                                        contiguous().view(-1, 2).to(device)).mean() + \
                                        torch.mean(torch.abs(dec_inp[:,1:,2]))
            batch_test_loss += test_loss.item()

            # calculating the position for each time step of prediction based on velocity
            preds_tr_b = (dec_inp[:, 1:, 0:2]*std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + \
                data['src'][:,-1:,0:2].cpu().numpy()
            pr.append(preds_tr_b)
            testing_loss.append(batch_test_loss/len(test_loader))

        # calculating mad and fad evaluation metrics
        gt = np.concatenate(gt, 0)
        pr = np.concatenate(pr, 0)
        obs = np.concatenate(obs, 0)
        mad, fad, _ = dataloader.distance_metrics(gt, pr)
        test_mad.append(mad)
        test_fad.append(fad)
    print(f"Average Mean Displacement: {np.mean(np.array(mad))}     Final Displacement Error: {np.mean(np.array(fad))}")
    # plotting the predicted and ground truth trajectories
    if PLOT_GRAPHS:
        folder_path = "visualize_it"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for _ in range(NUM_OF_PREDICTION_TEST):
            idx = np.random.randint(0, gt.shape[0])
            plt.figure()
            plt.scatter(gt[idx,:,0],gt[idx,:,1], color='green', label="Ground truth")
            plt.scatter(pr[idx,:,0],pr[idx,:,1], color='orange',label="Predictions")
            plt.scatter(obs[idx,:,0], obs[idx,:,1], color='b', label="Observations")
            plt.legend()
            plt.xlim(-8, 18)
            plt.ylim(-11, 15)
            plt.title("Trajectory Visualization in camera frame")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.savefig("{}/traj_{}.png".format(folder_path, idx))
            # plt.savefig("traj_{}".format(idx))
            
            plt.show()