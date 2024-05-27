import jax
import jax.numpy as jnp
import jax.random as jrng
import optax
from ray import tune
import random
import numpy as np
import pickle
import os
import sys
import gridworld
import policies as policies_module
import value_and_policy_iteration as vpi

from util import *
from main import *

from itertools import *

np.random.seed(42);
random.seed(42)


def compare_pve_mle(config):
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # construct env and get env params
    env = gridworld.FourRooms(p_intended=0.8)
    true_r = env.get_reward_matrix()
    true_p = env.get_transition_tensor()
    true_m = true_r, true_p
    num_states, num_actions = np.shape(true_r)

    num_ve_steps, ve_policy_mode = config['ve_mode'] 

    # collect policies and values 
    if not config['use_vip']:
        policies, values = policies_module.collect_random_policies(
            config['n_policies'], ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])
    else:
        policies, values = policies_module.collect_iteration_policies(
            1000, 100, ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])

    # print(f'{policies.shape=}')
    # print(f'{num_states=}')
    # print(f'{num_actions=}')

    split = int(config['policy_dataset_split'] * config['n_policies'])
    policies_train = policies[:split]
    policies_test = policies[split:]

    values_train = values[:split]
    values_test = values[split:]

    # fit models
    m_pve = fit_model_pve(policies_train, values_train, true_m, config)
    # m_mle = fit_model_mle(policies_train, true_m, config)

    # evaluate models
    eval_model_pred_err(policies_test, values_test, m_pve, config)


def eval_model_pred_err(policies, values, m, config):
    r, p = m
    pred_err = np.mean([np.abs(vpi.exact_policy_evaluation(config['gamma'], pi, r, p) - v_pi)
                        for pi, v_pi in zip(policies, values)])
    return pred_err


def fit_model_pve(policies, values, true_m, config):

    true_r, true_p = true_m
    num_states, num_actions = np.shape(true_r)

    # initialize jax stuff
    key = jrng.PRNGKey(config['seed'])
    key, model_params = init_model(key, num_states, num_actions, config)
    opt = optax.adam(config['learning_rate'])
    state = opt.init(model_params)

    num_ve_steps, ve_policy_mode = config['ve_mode'] 

    def _update_ve(params, state, pi, v):
        return update_ve(params, state, opt, pi, v, true_r, true_p, config)
    _update_ve = jax.jit(_update_ve)

    def _update_fpve(params, state, pi, true_v_pi):
        return update_fpve(params, state, opt, pi, true_v_pi, config)
    _update_fpve = jax.jit(_update_fpve)

    reports = []
    stored_models = []
    # stored_models_path = os.path.join(tune.get_trial_dir(), 'models.pickle')
    for ts in range(1, config['pve_n_iters']+1):
        idx = np.random.randint(0, len(policies), size=[config['batch_size']])
        pi_batch = policies[idx, :, :]
        v_batch = values[idx, :]
        if num_ve_steps == np.inf:
            model_params, state, loss = _update_fpve(model_params, state, pi_batch, v_batch)
        else:
            model_params, state, loss = _update_ve(model_params, state, pi_batch, v_batch)
        report = {}
        if ts % config['store_model_every'] == 0:
            r, p = params_to_model(model_params, config)
            stored_models.append((ts, r, p))
        if ts % config['store_loss_every'] == 0:
            report['loss'] = float(loss)
        if ts % config['eval_model_every'] == 0:
            r, p = params_to_model(model_params, config)
            r, p = np.array(r), np.array(p)
            _, pi = vpi.run_value_iteration(
                config['gamma'], r, p, np.zeros([num_states]), threshold=1e-4, return_policy=True)
            value_pi = vpi.exact_policy_evaluation(config['gamma'], pi, true_r, true_p)
            report['mean_value'] = np.mean(value_pi)
            report['ts'] = ts
        if ts % config['ping_every'] == 0:
            print(f'{ts=}')
        
        if len(stored_models) > 0 and ts == (config['num_iters'] - 1):
            with open(stored_models_path, 'wb') as f:
                pickle.dump(stored_models, f) 
            report['model_path'] = stored_models_path
        if len(report) > 0:
            # tune.report(**to_report)
            report['ts'] = ts
            reports.append(report)
    dump(reports, 'reports')

    return params_to_model(model_params, config)


def main2():

    every = 100

    space = {
        # FIXME
        # 'seed': tune.randint(0, 500_000),
        'seed': 0,
        'gamma': 0.99,
        'batch_size': 50,
        # 've_mode': tune.grid_search([(np.inf, 'stoch'), (np.inf, 'det')]),
        've_mode': (np.inf, 'det'),
        # FIXME
        # 'model_rank': tune.grid_search([20, 30, 40, 50, 60, 70, 80, 90, 100, 104]),
        'model_rank': 80,
        'learning_rate': 5e-4,
        'eval_model_every': 10_000,
        'store_loss_every': every,
        'restrict_capacity': True,
        'store_model_every': np.inf,
        'uni_init': 5,
        'use_vip': False,
        # ---
        # fixme
        'n_policies': 10,
        'policy_dataset_split': .5,
        'pve_n_iters': 10_000,
        'mle_n_iters': 30_000,
        'ping_every': every,
    }

    # local_dir, seed = sys.argv[1:]
    # seed = int(seed)
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    num_samples = 10
    # analysis = tune.run(run_experiment_2,
    #                     num_samples=num_samples,
    #                     config=space,
    #                     local_dir=local_dir,
    #                     resources_per_trial={'cpu': 1},
    #                     fail_fast=True)

    compare_pve_mle(space)


if __name__ == '__main__':
    main2()

