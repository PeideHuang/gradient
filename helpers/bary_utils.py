import numpy as np
import ot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from helpers.w_encode import *
from sklearn import preprocessing
from matplotlib.patches import Rectangle
from typing import List
import os

reg = 10  # Entropic Regularization
numItermax = 50  # Maximum number of iterations for the Barycenter algorithm
numInnerItermax = 100  # Maximum number of sinkhorn iterations
n_samples = 100  # number of samples for barycenter

# visualize the continuous maze environment
def visualize_V_2d(model, fig_save_path, fig, ax):
    x = np.linspace(-6, 14, 100)
    y = np.linspace(-6, 14, 100)
    X, Y = np.meshgrid(x, y)
    C = np.stack((X, Y), axis=-1).reshape(-1, 2)
    S = np.concatenate([np.zeros((C.shape[0], 7)), C], axis=1)
    if model is not None:
        V = compute_V(S, model).reshape(100, 100)
        c = ax.pcolormesh(x, y, V, cmap='RdBu_r')
    map = Map()
    ax.plot(map.ox, map.oy, ".k")
    # fig.colorbar(c, orientation='vertical')
    ax.set(xlim=[-6, 14], ylim=[-6, 14])
    ax.set_aspect('equal', 'box')
    # plt.savefig(fig_save_path)
    # plt.show()
    # plt.close()
    return fig, ax

def compute_barycenter(source, target, n_samples, alpha):
    s, t = source, target
    assert s.shape[1] == t.shape[1]

    d = s.shape[1]

    # print('Original dimension: ', s.shape[1])
    # print('Source data size: ', s.shape)
    # print('Target data size: ', t.shape)
    # print('Barycenter sample size: ', n_samples)
    # print('alpha: ', alpha)

    a1, a2 = ot.unif(len(s)), ot.unif(len(t))

    XB_init = np.random.randn(n_samples, d)

    XB = ot.lp.free_support_barycenter(
        measures_locations=[s, t],
        measures_weights=[a1, a2],
        weights=np.array([1-alpha, alpha]),
        X_init=XB_init,
        numItermax=numItermax,
        verbose=False
    )

    return XB

def compute_V(s, model):
    return model.predict(s[:, -2:])


class Distance(object):
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def dist_func(self, g1, g2):
        g1 = g1*self.scaler.scale_ + self.scaler.mean_
        g2 = g2*self.scaler.scale_ + self.scaler.mean_

        n = g1.shape[0]

        s1 = np.concatenate([np.zeros((n, 7)), g1], axis=1)
        s2 = np.concatenate([np.zeros((n, 7)), g2], axis=1)

        return np.abs(compute_V(s1, self.model) - compute_V(s2, self.model))


class GRADIENT(object):
    def __init__(self, encoder_input_dim, embedding_dim=2, beta=5.0, interp_metric='l2'):
        self.cae = CAE(encoder_input_dim, embedding_dim, beta=beta)
        self.interp_metric = interp_metric

    def compute_barycenter(        
        self,
        source_dist: np.array,
        target_dist: np.array,
        alpha: float,
        model,
        fig_save_path,
        low,
        high,
        N = 150,
        N_bary_samples = 20,
        epoch = 100,
        batch_size = 512,
        visualize = False,):
        if self.interp_metric == 'l2':
            return self.compute_barycenter_l2(
                source_dist,
                target_dist,
                alpha,
                fig_save_path,
                low,
                high,
                N_bary_samples,
                visualize,
            )
        elif self.interp_metric == 'encoding':
            return self.train_CAE_and_compute_barycenter(
                    source_dist,
                    target_dist,
                    alpha,
                    model,
                    fig_save_path,
                    low,
                    high,
                    N,
                    N_bary_samples,
                    epoch,
                    batch_size,
                    visualize,
            )
        else:
            raise NotImplementedError('interp_metric must be one of [''l2'', ''encoding'']!')

    def compute_barycenter_l2(self, 
                              source_dist, 
                              target_dist, 
                              alpha, 
                              fig_save_path, 
                              low, 
                              high, 
                              N_bary_samples, 
                              visualize,
                              ):

        # only for visualizing purpose
        bary_centers = []
        n_alphas = 11
        alphas = np.linspace(0, 1, n_alphas)
        for a in alphas:
            bary_center = compute_barycenter(source_dist, target_dist, N_bary_samples, a)   
            bary_centers.append(bary_center)

        bary_center_return = compute_barycenter(source_dist, target_dist, N_bary_samples, alpha)   

        # plot
        if visualize:
            fig, ax = plt.subplots()
            fig, ax = visualize_V_2d(None, fig_save_path + '_vis_V', fig, ax)

            cmap = plt.get_cmap('cool')
            for i in range(n_alphas):
                rgb = tuple(np.array(cmap(float(alphas[i]))[:3]))
                ax.scatter(x=bary_centers[i][:, 0], y=bary_centers[i][:, 1], color=rgb, edgecolor='k', label='Barycenter', alpha=0.6)
            ax.scatter(x=bary_center_return[:, 0], y=bary_center_return[:, 1], color='yellow', edgecolor='k', label='Highlighted')
            ax.set(xlim=[-6, 14], ylim=[-6, 14])
            ax.set_aspect('equal', 'box')
            plt.savefig(fig_save_path)
            plt.close()

        return bary_center_return

    def train_CAE_and_compute_barycenter(
        self,
        source_dist: np.array,
        target_dist: np.array,
        alpha: float,
        model,
        fig_save_path,
        low,
        high,
        N = 150,
        N_bary_samples = 20,
        epoch = 100,
        batch_size = 512,
        visualize = False,
        # highlight_idx = None,
    ):
        print('Training CAE...')
        # data_x = -2 + 12*np.random.rand(N, 1)
        # data_y = -2 + 12*np.random.rand(N, 1)
        # data = np.concatenate([data_x, data_y], axis=1)
        data = np.random.uniform(low=low, high=high, size=(N, source_dist.shape[1]))
        # source = np.array([[8, 0]]) + np.random.randn(20, 2)
        # target = np.array([[8, 8]]) + np.random.randn(20, 2)
        
        # transform dataset
        scaler = preprocessing.StandardScaler().fit(data)
        scaler.scale_ = 1
        scaler.mean_ = 0

        data_transformed = scaler.transform(data)
        source_transformed = scaler.transform(source_dist)
        target_transformed = scaler.transform(target_dist)
        distance = Distance(scaler, model)

        # print('scaler.scale_, scaler.mean_: ', scaler.scale_, scaler.mean_)
        
        self.cae.prepare_dataset(data_transformed, distance.dist_func)
        self.cae.train(epochs=epoch, plot_loss=False, batch_size=batch_size)
        # cae.save_model()
        # cae.test()
        # cae.load_model()

        source_encoded = self.cae.encoder(self.cae.numpy_to_cuda(source_transformed)).cpu().detach().numpy()
        target_encoded = self.cae.encoder(self.cae.numpy_to_cuda(target_transformed)).cpu().detach().numpy()

        # only for visualizing purpose
        bary_centers = []
        n_alphas = 11
        alphas = np.linspace(0, 1, n_alphas)
        for a in alphas:
            bary_center_encoded = compute_barycenter(source_encoded, target_encoded, N_bary_samples, a)   
            bary_center_decoded = self.cae.decoder(self.cae.numpy_to_cuda(bary_center_encoded)).cpu().detach().numpy()

            # rescale back the data
            bary_center_decoded = bary_center_decoded*scaler.scale_ + scaler.mean_
            bary_centers.append(bary_center_decoded)

        # compute the barycenter
        bary_center_return_encoded = compute_barycenter(source_encoded, target_encoded, N_bary_samples, alpha)   
        bary_center_return_decoded = self.cae.decoder(self.cae.numpy_to_cuda(bary_center_return_encoded)).cpu().detach().numpy()

        # rescale back the data
        bary_center_decoded = bary_center_decoded*scaler.scale_ + scaler.mean_
        bary_center_return_decoded = bary_center_return_decoded*scaler.scale_ + scaler.mean_

        # test: linear interpolation
        # s = np.array([[0., 0.]])
        # t = np.array([[0., 8.]])
        # s_transformed = scaler.transform(s)
        # t_transformed = scaler.transform(t)
        # s_encoded = cae.encoder(cae.numpy_to_cuda(s_transformed)).cpu().detach().numpy()
        # t_encoded = cae.encoder(cae.numpy_to_cuda(t_transformed)).cpu().detach().numpy()
        # traj = []
        # for alpha in np.linspace(0, 1, 20):
        #     inter = (1 - alpha)*s_encoded + alpha*t_encoded
        #     inter_decoded = cae.decoder(cae.numpy_to_cuda(inter)).cpu().detach().numpy()
        #     inter_decoded = inter_decoded*scaler.scale_ + scaler.mean_
        #     traj.append(inter_decoded)
        # traj = np.array(traj)

        # plot
        if visualize:
            fig, ax = plt.subplots()
            fig, ax = visualize_V_2d(model, fig_save_path + '_vis_V', fig, ax)

            cmap = plt.get_cmap('cool')
            for i in range(n_alphas):
                rgb = tuple(np.array(cmap(float(alphas[i]))[:3]))
                ax.scatter(x=bary_centers[i][:, 0], y=bary_centers[i][:, 1], color=rgb, edgecolor='k', label='Barycenter', alpha=0.6)
            ax.scatter(x=bary_center_return_decoded[:, 0], y=bary_center_return_decoded[:, 1], color='yellow', edgecolor='k', label='Highlighted')
            ax.set(xlim=[-6, 14], ylim=[-6, 14])
            ax.set_aspect('equal', 'box')
            plt.savefig(fig_save_path)
            plt.close()
        
        print('-'*50)
        return bary_center_return_decoded


class Map(object):
    def __init__(self) -> None:
        self.ox, self.oy = [], []
        for i in range(-2, 10):
            self.ox.append(i)
            self.oy.append(-2)
        for i in range(-2, 11):
            self.ox.append(10.0)
            self.oy.append(i)
        for i in range(-2, 10):
            self.ox.append(i)
            self.oy.append(10.0)
        for i in range(-2, 10):
            self.ox.append(-2)
            self.oy.append(i)

        for i in range(-2, 6):
            self.ox.append(i)
            self.oy.append(2)
        for i in range(2, 7):
            self.ox.append(6.0)
            self.oy.append(i)
        for i in range(-2, 6):
            self.ox.append(i)
            self.oy.append(6)