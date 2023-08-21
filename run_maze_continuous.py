import argparse
from collections import deque
import gym
from gym.envs.registration import register
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.time_limit import TimeLimit
import os
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from matplotlib import cm
import mujoco_maze
from helpers.bary_utils import *
from helpers.custom_callback import StopTrainingCallback, StopOnMaxTimestepsCallback
from helpers.utils import make_vec_env, linear_schedule, Reward_Estimation_Model, wrap_vec_env, set_context_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="maze", help='Experiment name prefix')
    parser.add_argument("--curriculum", type=str, default="gradient", choices=["no_curr", "uniform", "gradient"])
    parser.add_argument('--interp_metric', type=str, default="l2", choices=["encoding", "l2"])
    parser.add_argument('--num_stage', type=int, default=10, help='Number of stages')
    parser.add_argument('--reward_threshold', type=float, default=0.5, help='Reward threshold for GRADIENT')
    parser.add_argument('--stage_max_step', type=int, default=100000, help='Maximum steps per stage')
    parser.add_argument('--exploration_step', type=int, default=10000, help='Number of uniform samples per stage for exploration')
    parser.add_argument('--gp_buffer_size', type=int, default=200, help='Buffer size of gaussian process regression')
    parser.add_argument('--eval_mode', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--learner', type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    # seed all randomness
    seed = args.seed
    set_random_seed(seed)
    torch.manual_seed(seed)

    # set context upper and lower limits here
    context_low = [-2, -2]
    context_high = [10, 10]

    # specify source and target empirical distribution here, represented by a list of samples
    source_dist = np.random.randn(100, len(context_high))
    target_dist = np.array([[0, 8]]) + 1*np.random.randn(100, len(context_high))

    verbose = 1
    total_timesteps = 1e6
    n_train_env = 16 # num of parallel environments
    n_eval_env = 8

    reward_threshold = args.reward_threshold

    n_steps = int(1000/n_train_env)
    eval_freq = n_steps*5

    exp_name = args.exp_name    + "_" + args.curriculum \
                                + "_metric_" + args.interp_metric \
                                + "_rt_" + str(reward_threshold).replace(".", "_" ) \
                                + '_stagestep_' + str(args.stage_max_step) \
                                + '_nstage_' + str(args.num_stage) \
                                + '_expl_' + str(args.exploration_step) \
                                + '_gp_buffer_size_' + str(args.gp_buffer_size) \
                                + "_seed_" + str(seed)
    
    # easiest task
    env = 'PointUMaze-v1'
    log_path = "./logs/" + env + "/"
    env_min_return = -1. # minimum return for the environment

    # create training environment
    train_env = make_vec_env(env, n_envs=n_train_env, seed=seed, vec_env_cls=SubprocVecEnv)

    # create curriculum evaluation environment
    cur_env = make_vec_env(env, n_envs=n_eval_env, seed=seed, vec_env_cls=SubprocVecEnv)

    # create evaluation environment
    eval_env = make_vec_env(env, n_envs=n_eval_env, seed=seed, vec_env_cls=SubprocVecEnv)
    set_context_dist(eval_env, target_dist)

    # create evaluation callback functions
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path + exp_name+'_0/',
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_env,
        deterministic=False,
        render=False
    )

    # checkpoint_callback = CheckpointCallback(save_freq=eval_freq, 
    #                                 save_path=log_path + exp_name+'_0/',
    #                                 verbose=1)

    # PPO hyperparameters
    if args.learner == "ppo":
        batch_size = 256
        gamma = 0.99
        gae_lambda = 0.95
        n_epochs = 20
        ent_coef = 0.01
        max_grad_norm = 0.5
        vf_coef = 0.5
        lr = 3e-4
        clip_range = 0.2
        policy_kwargs = dict(net_arch = [dict(pi=[128, 128], vf=[128, 128])])
        target_kl = None

        model = PPO('MlpPolicy', train_env, verbose=0, learning_rate=lr, batch_size=batch_size, 
                    gamma=gamma, gae_lambda=gae_lambda, n_epochs=n_epochs, max_grad_norm=max_grad_norm,
                    ent_coef=ent_coef, vf_coef=vf_coef, clip_range=clip_range, 
                    n_steps=n_steps, tensorboard_log=log_path, 
                    target_kl=target_kl, policy_kwargs=policy_kwargs, seed=seed)
    # SAC hyperparameters
    elif args.learner == "sac":
        buffer_size = int(1e5)
        ent_coef = 'auto'
        target_entropy="auto"
        batch_size = 256
        gamma = 0.99
        lr = 1e-3
        learning_starts = 1000
        train_freq = 1
        gradient_steps = 1
        policy_kwargs = dict(net_arch = [64, 64, 64], activation_fn = torch.nn.Tanh)
        model = SAC('MlpPolicy', train_env, verbose=0, seed=seed,
                train_freq=train_freq, gradient_steps=gradient_steps, buffer_size=buffer_size, ent_coef=ent_coef,
                gamma=gamma, learning_rate=linear_schedule(lr), learning_starts=learning_starts, 
                policy_kwargs=policy_kwargs, batch_size=batch_size, target_entropy=target_entropy,
                tensorboard_log=log_path)
    else:
        raise NotImplementedError('Learner not implemented')

    if args.eval_mode:
        assert args.model_path is not None
        print('Load model from ', args.model_path)
        model = PPO.load(args.model_path)

        evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False, render=True)
        return

    print('******************** Start Curriculum Training ********************')
    if args.curriculum == "no_curr":
        print('-'*30)
        print('Method: No Curriculum')
        print('-'*30)
        set_context_dist(train_env, target_dist)

        model.learn(total_timesteps=total_timesteps, 
                    tb_log_name=exp_name, 
                    callback=[eval_callback], 
                    reset_num_timesteps=False)
    elif args.curriculum == "uniform":
        print('-'*30)
        print('Method: Uniform Domain Randomization')
        print('-'*30)

        uniform_dist = np.random.uniform(low=context_low, high=context_high, size=(500, len(context_high)))
        set_context_dist(train_env, uniform_dist)

        model.learn(total_timesteps=total_timesteps, 
                    tb_log_name=exp_name, 
                    callback=[eval_callback], 
                    reset_num_timesteps=False)
    elif args.curriculum == "gradient":
        print('-'*30)
        print('Method: GRADIENT')
        print('-'*30)

        max_stage = args.num_stage
        
        # two different way of defining alphas
        # alphas = np.linspace(0., 1., max_stage+1)
        alphas = [0]
        for i in range(max_stage):
            alphas.append(1/(max_stage - i))

        print('alphas: ', alphas)
        gradient = GRADIENT(source_dist.shape[1], beta=10., interp_metric=args.interp_metric)

        if args.interp_metric == "encoding":
            r_est_model = Reward_Estimation_Model(env_min_return, 0.2, buffer_size=args.gp_buffer_size)
        else:
            r_est_model = None
        
        max_stage += 1
        for i in range(max_stage):
            print('*'*30)
            print('GRADIENT Stage: ', i)
            print('*'*30)

            # inter_dist = source_dist
            fig_save_folder = 'visualization/'+ exp_name 
            os.makedirs(fig_save_folder, exist_ok=True) 
            fig_save_path = fig_save_folder + '/PointUMaze_' + str(i)

            if i == 0:
                inter_dist = np.copy(source_dist)
            else:
                temp_source = np.copy(inter_dist)
                inter_dist = np.copy(gradient.compute_barycenter(
                                                    temp_source,
                                                    target_dist, 
                                                    alphas[i],
                                                    r_est_model, 
                                                    fig_save_path,
                                                    context_low,
                                                    context_high,
                                                    visualize=True,
                                                    N_bary_samples=100,
                                                    epoch=100, # 100
                                                    N=500, 
                                                ))

            # set context distribution
            set_context_dist(train_env, inter_dist)
            set_context_dist(cur_env, inter_dist)
            
            # if last stage, do not stop training
            rt = 1e10 if i == (max_stage-1) else reward_threshold
            threshold_callback = StopTrainingOnRewardThreshold(rt, verbose=verbose)
            stop_training_callback = StopTrainingCallback(
                                                            cur_env,
                                                            callback_on_new_best = threshold_callback,
                                                            log_path=log_path,
                                                            eval_freq=eval_freq,
                                                            n_eval_episodes=n_eval_env,
                                                            deterministic=False,
                                                            render=False,
                                                        )
            # to deal with stable-baselines3 convention of counting steps
            if i != (max_stage-1):                                             
                stage_max_stop_step = model.num_timesteps + args.stage_max_step
            else:
                stage_max_stop_step = total_timesteps
            stop_on_max_timesteps_callback = StopOnMaxTimestepsCallback(stage_max_stop_step)
            model.set_env(train_env)
            print('Start to train in the current stage: ', i)
            
            # policy learning phase
            model.learn(total_timesteps=int(total_timesteps - model.num_timesteps), 
                        tb_log_name=exp_name, 
                        callback=[eval_callback, stop_training_callback, stop_on_max_timesteps_callback], 
                        # callback=[eval_callback, stop_on_max_timesteps_callback], 
                        reset_num_timesteps=False)

            # exploration + train GP (only for encoding metric)
            if args.interp_metric == "encoding":
                print('Exploration on uniform task distribution.')
                stop_on_max_timesteps_callback = StopOnMaxTimestepsCallback(args.exploration_step + model.num_timesteps)

                exploration_dist = np.random.uniform(low=context_low, high=context_high, size=(500, len(context_low)))
                set_context_dist(train_env, exploration_dist)

                model.set_env(train_env)
                model.learn(total_timesteps=int(total_timesteps - model.num_timesteps), 
                            tb_log_name=exp_name, 
                            callback=[eval_callback, stop_on_max_timesteps_callback], 
                            reset_num_timesteps=False)

                # train gaussian process
                explore_c, explore_r = [epi_info["context"] for epi_info in model.ep_info_buffer], [epi_info["r"] for epi_info in model.ep_info_buffer]
                # explore_c, explore_r = np.array(explore_c), np.array(explore_r)
                print('Exploration Episodes Num: ', len(explore_r))

                data_c, data_r = explore_c, explore_r
                r_est_model.train(data_c, data_r)
                    
            model.logger.record("curriculum/current_stage", i+1)
            model.logger.dump(model.num_timesteps)        
            # print('Finished training in the current stage: ', i)
            # model.save("models/"+exp_name+"_model")
    else:
        raise NotImplementedError('Curriculum not implemented')

if __name__ == "__main__":
    main()