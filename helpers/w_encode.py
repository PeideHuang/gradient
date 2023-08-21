# from base64 import encode
# from json import encoder
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
import itertools
from tqdm import tqdm

from scipy.spatial import distance

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = 128

        self.model = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features=kwargs["embedding_dim"]),
        )

    def forward(self, c):
        return self.model(c)

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = 128

        self.model = nn.Sequential(
            nn.Linear(in_features=kwargs["embedding_dim"], out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features=kwargs["output_shape"]),
        )

    def forward(self, z):
        return self.model(z)

class PrepareData(Dataset):
    def __init__(self, X, dist_func, maximum_num_data=250000):
        self.dist_func = dist_func
        self.data = []
        for combination in itertools.product(X, repeat=2):
            if not np.array_equal(combination[0], combination[1]):
                self.data.append((combination[0], combination[1])) #, self.dist_func(combination[0], combination[1])
                # print(self.data[-1][-1])
        self.data = np.array(self.data)
        # print(self.data.shape)
        self.data = self.data[:maximum_num_data, :, :]
        self.distance = self.dist_func(self.data[:, 0, :], self.data[:, 1, :])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return c1, c2, dist_encoding
        return self.data[idx, 0, :], self.data[idx, 1, :], self.distance[idx]


def l2_func(x1, x2):
    return np.linalg.norm(x1-x2)
def cos(x1, x2):
    return distance.cosine(x1, x2)
def cityblock(x1, x2):
    return distance.cityblock(x1, x2)

# Contextual Autoencoder
class CAE(object):
    def __init__(self, input_dim, embedding_dim, beta=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta

        # make dataset
        # self.prepare_dataset(c, dist_func)

        self.encoder = Encoder(input_shape=input_dim, embedding_dim=embedding_dim).to(self.device)
        self.decoder = Decoder(output_shape=input_dim, embedding_dim=embedding_dim).to(self.device)

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        self.recon_criterion = nn.MSELoss()

        # # mean-squared error loss

    def prepare_dataset(self, c, dist_func, test_split=.2):
        print('Preparing dataset...')
        dataset = PrepareData(c, dist_func)
        print('Finished preparing dataset. len(dataset): ', len(dataset))

        dataset_size = len(dataset)
        split = int(np.floor(test_split * dataset_size))
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [dataset_size - split, split])
        # print(len(dataset), len(train_dataset), len(test_dataset))


    def loss_func(self, c1, c2, dist, print_loss=False):

        z1, z2 = self.encoder(c1), self.encoder(c2)
        r1, r2 = self.decoder(z1), self.decoder(z2)

        # reconstruction loss
        recon_loss = self.recon_criterion(c1, r1) + self.recon_criterion(c2, r2)

        # distance loss
        # print(torch.norm(z1 - z2, dim=1).shape, dist.shape)
        dist_loss = self.recon_criterion(torch.norm(z1 - z2, dim=1), dist)
        # print(c1[0], r1[0], c2[0], r2[0], dist[0])
        if print_loss:
            print('recon_loss: ', recon_loss.item(), ' beta*dist_loss: ', self.beta*dist_loss.item())

        return recon_loss, self.beta*dist_loss

    def numpy_to_cuda(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def train(self, epochs = 200, learning_rate = 1e-3, plot_loss=False, batch_size = 512):
        self.encoder.train()
        self.decoder.train()
        self.optimizer = optim.Adam(self.params, lr=learning_rate)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        loss_epoch = []
        recon_loss_epoch = []
        distance_loss_epoch = []
        for epoch in range(epochs):
            loss = 0.
            recon_loss_epoch = 0.
            distance_loss_epoch = 0.
            for c1, c2, dist in self.train_loader:
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                # batch_features = batch_features.view(-1, 784).to(device)
                c1, c2, dist = c1.float().to(self.device), c2.float().to(self.device), dist.float().to(self.device)
                
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                self.optimizer.zero_grad()
                
                # compute training reconstruction loss
                recon_loss, distance_loss = self.loss_func(c1, c2, dist)
                train_loss = recon_loss + distance_loss
                
                # compute accumulated gradients
                train_loss.backward()
                
                # perform parameter update based on current gradients
                self.optimizer.step()
                
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                recon_loss_epoch += recon_loss.item()
                distance_loss_epoch += distance_loss.item()
            
            # compute the epoch training loss
            loss = loss / len(self.train_loader)
            recon_loss_epoch = recon_loss_epoch / len(self.train_loader)
            distance_loss_epoch = distance_loss_epoch / len(self.train_loader)
            loss_epoch.append(loss)
            # display the epoch training loss
            if epoch % 10 == 0:
                print("epoch : {}/{}, training loss = {:.8f}, recon loss = {:.8f}, distance loss = {:.8f}".format(epoch, epochs, loss, recon_loss_epoch, distance_loss_epoch))
                self.test(batch_size)

        if plot_loss:
            plt.figure()
            plt.plot(range(epochs), loss_epoch)
            plt.savefig('training_loss')
            plt.close()

    def test(self, batch_size):
        self.encoder.eval()
        self.decoder.eval()

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True
        )

        loss_total = 0.
        recon_loss_total = 0.
        distance_loss_total= 0.
        for c1, c2, dist in self.test_loader:
            c1, c2, dist = c1.float().to(self.device), c2.float().to(self.device), dist.float().to(self.device)
            # z1, z2 = self.encoder(c1), self.encoder(c2)
            # r1, r2 = self.decoder(z1), self.decoder(z2)

            recon_loss, distance_loss = self.loss_func(c1, c2, dist)
            test_loss = recon_loss + distance_loss

            loss_total += test_loss.item()
            recon_loss_total += recon_loss.item()
            distance_loss_total += distance_loss.item()

            # print('-'*50)
            # print('Reconstruction: ')
            # print(c1.cpu().detach().numpy(), r1.cpu().detach().numpy())
            # print(c2.cpu().detach().numpy(), r2.cpu().detach().numpy())
            # print('Encoding distance:')
            # print(dist.item(), torch.norm(z1 - z2, dim=1).item())
        loss = loss_total / len(self.test_loader)
        recon_loss = recon_loss_total / len(self.test_loader)
        distance_loss = distance_loss_total / len(self.test_loader)
        print("Test: loss = {:.8f}, recon loss = {:.8f}, distance loss = {:.8f}".format(loss, recon_loss, distance_loss))
        # print('Test loss: ', test_loss.item())

        self.encoder.train()
        self.decoder.train()

    def save_model(self):
        torch.save(self.encoder.state_dict(), 'encoder.pth')
        torch.save(self.decoder.state_dict(), 'decoder.pth')

    def load_model(self):
        self.encoder.load_state_dict(torch.load('encoder.pth'))
        self.decoder.load_state_dict(torch.load('decoder.pth'))

# if __name__ == "__main__":
#     embedding_dim = 128

#     batch_size = 512
#     epochs = 200
#     learning_rate = 1e-3
#     d = 2
#     N_train = 200
#     test_split = .01

#     # make dataset
#     c = np.random.rand(N_train, d)
#     print('Preparing dataset')
#     dataset = PrepareData(c, cityblock)
#     print('Finished preparing dataset')

#     dataset_size = len(dataset)
#     # indices = list(range(dataset_size))
#     split = int(np.floor(test_split * dataset_size))
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [dataset_size - split, split])
#     # print(len(dataset), len(train_dataset), len(test_dataset))

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True
#     )

#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=1, shuffle=True
#     )

#     # #  use gpu if available
#     # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")

#     # # create a model from `AE` autoencoder class
#     # # load it to the specified device, either gpu or cpu
#     encoder = Encoder(input_shape=d, embedding_dim=embedding_dim).to(device)
#     decoder = Decoder(output_shape=d, embedding_dim=embedding_dim).to(device)

#     # # create an optimizer object
#     # # Adam optimizer with learning rate 1e-3
#     params = list(encoder.parameters()) + list(decoder.parameters())
#     optimizer = optim.Adam(params, lr=learning_rate)

#     # # mean-squared error loss
#     recon_criterion = nn.MSELoss()

#     for epoch in range(epochs):
#         loss = 0
#         for c1, c2, dist in train_loader:
#             # reshape mini-batch data to [N, 784] matrix
#             # load it to the active device
#             # batch_features = batch_features.view(-1, 784).to(device)
#             c1, c2, dist = c1.float().to(device), c2.float().to(device), dist.float().to(device)
            
#             # reset the gradients back to zero
#             # PyTorch accumulates gradients on subsequent backward passes
#             optimizer.zero_grad()
            
#             # compute training reconstruction loss
#             train_loss = loss_func(c1, c2, dist)
            
#             # compute accumulated gradients
#             train_loss.backward()
            
#             # perform parameter update based on current gradients
#             optimizer.step()
            
#             # add the mini-batch training loss to epoch loss
#             loss += train_loss.item()
        
#         # compute the epoch training loss
#         loss = loss / len(train_loader)
        
#         # display the epoch training loss
#         print("epoch : {}/{}, training loss = {:.8f}".format(epoch + 1, epochs, loss))

#     encoder.eval()
#     decoder.eval()

#     for c1, c2, dist in test_loader:
#         # reshape mini-batch data to [N, 784] matrix
#         # load it to the active device
#         # batch_features = batch_features.view(-1, 784).to(device)
#         c1, c2, dist = c1.float().to(device), c2.float().to(device), dist.float().to(device)

#         z1, z2 = encoder(c1), encoder(c2)
#         r1, r2 = decoder(z1), decoder(z2)

#         print('-'*50)
#         print('Reconstruction: ')
#         print(c1, r1)
#         print(c2, r2)
#         print('Encoding distance:')
#         print(dist.item(), torch.norm(z1 - z2, dim=1).item())
        
#         # compute training reconstruction loss
#         train_loss = loss_func(c1, c2, dist)
        
#         # add the mini-batch training loss to epoch loss
#         # loss += train_loss.item()

#         # print(train_loss.item())