import os
from typing import Any, Callable, Dict, Optional, Type, Union, List
import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from helpers.monitor import CustomMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from matplotlib import cm
import torch
import gpytorch
from helpers.bary_utils import *
import numpy as np

# NOTE: must implement set_context_dist function in the environment
def set_context_dist(venv, context_dist, vectorize=True):
    if vectorize:
        venv.env_method('set_context_dist', context_dist)
    else:
        for e in venv.envs:
            e.set_context_dist(context_dist)

def wrap_vec_env(venv, wrapper, wrapper_kwargs):
    wrapped = []    
    for e in venv.envs:
        e = wrapper(e, **wrapper_kwargs)
        wrapped.append(e)
    venv.envs = wrapped
    return venv

def make_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[List[Callable[[gym.Env], gym.Env]]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[List[Dict[str, Any]]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = CustomMonitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                for i, wrapper in enumerate(wrapper_class):
                    env = wrapper(env, **(wrapper_kwargs[i]))
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

from typing import Callable
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value + 1e-4

    return func

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, prior_mean, prior_std):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ConstantMean(gpytorch.priors.NormalPrior(prior_mean, prior_std))

        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior = gpytorch.priors.GammaPrior(2.0, 2.0)))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

from collections import deque
class Reward_Estimation_Model():
    def __init__(self, prior_mean, prior_std, buffer_size=500) -> None:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.train_x_buffer = deque(maxlen=buffer_size)
        self.train_y_buffer = deque(maxlen=buffer_size)
    
    def train(self, train_x, train_y):
        # append train_x and train_y to buffer
        self.train_x_buffer.extend(train_x)
        self.train_y_buffer.extend(train_y)

        print('-'*50)
        print("GP training data size: ", len(self.train_x_buffer))
        train_x = np.array(self.train_x_buffer)
        train_y = np.array(self.train_y_buffer)

        train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
        self.model = ExactGPModel(train_x, train_y, self.likelihood, self.prior_mean, self.prior_std)

        training_iter = 100

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if i % 10 == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            optimizer.step()
        
    
    def predict(self, test_x):
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.Tensor(test_x)

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))

        return observed_pred.mean.numpy()