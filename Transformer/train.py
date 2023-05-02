import torch
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import os
import dataloader 
from Model.transformer import Transformer
import matplotlib.pyplot as plt
from Model.utils import subsequent_mask
if __name__ == "__main__":

    val_size = 0
    observation = 8
    prediction = 12
    batch_size = 64
    epochs = 100
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_dataset, _ = dataloader.create_dataset('./dataset', 'raw', val_size,observation, prediction, delim="\t", train=True)
    val_dataset, _ = dataloader.create_dataset('./dataset', 'raw', val_size, observation, prediction, delim="\t", train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)

    mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
    std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(
            torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)

    encoder_ip_size = 2
    decoder_ip_size = 3
    model_op_size = 3
    emb_size = 512
    num_heads = 8
    ff_hidden_size = 2048
    n = 6
    dropout=0.1

    tf_model = Transformer(encoder_ip_size, decoder_ip_size, model_op_size, emb_size, num_heads, ff_hidden_size, n, dropout=0.1).to(device)

    optimizer = torch.optim.SGD(tf_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

    training_loss = []
    validation_loss = []
    val_mad = []
    val_fad = []

    T = epochs * len(train_loader)

    for epoch in tqdm(range(epochs)):

        tf_model.train()
        train_batch_loss = 0

        for idx, data in enumerate(train_loader):

            enc_input = (data['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)

            target = (data['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
            target_append = torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            target = torch.cat((target,target_append),-1)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
            dec_input = torch.cat((start_of_seq, target), 1)

            dec_source_mask = torch.ones((enc_input.shape[0], 1,enc_input.shape[1])).to(device)
            dec_target_mask = subsequent_mask(dec_input.shape[1]).repeat(dec_input.shape[0],1,1).to(device)

            optimizer.zero_grad()
            predictions = tf_model.forward(enc_input, dec_input, dec_source_mask, dec_target_mask)

            loss = F.pairwise_distance(predictions[:, :,0:2].contiguous().view(-1, 2),
                                       ((data['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).\
                                        contiguous().view(-1, 2).to(device)).mean() + \
                                        torch.mean(torch.abs(predictions[:,:,2]))
            
            train_batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        training_loss.append(train_batch_loss/len(train_loader))
        print("Epoch {}/{}....Training loss = {:.4f}".format(epoch+1, epochs, training_loss[-1]))

        torch.save({
            'model_state_dict': tf_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'val_mad': val_mad,
            'val_fad':val_fad,
            'learning_rate':lr
            }, os.path.join('./models', 'epoch{}.pth'.format(epoch+1)))


