import argparse
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import hssm
import ssms
import pymc as pm
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='model fitting')
    parser.add_argument('--run_index', type=int, required=True, help='index')
    return parser.parse_args()

def generate_simulation_data(n_samples):
    """Generates random parameter values and forward simulates from them"""
    true_theta_list = []
    sim_data_list = []

    for _ in range(n_samples):
        p_target = np.random.uniform(2, 5.5)
        theta = {
            'a': np.random.uniform(0.3, 3.0),
            'z': np.random.uniform(0.1, 0.9),
            't': np.random.uniform(0.001, 2),
            'p_target': p_target,
            'p_outer': random.choice([-1, 1]) * p_target,
            'p_inner': random.choice([-1, 1]) * p_target,
            'r': np.random.uniform(0.01, 0.05),
            'sda': np.random.uniform(1, 3),
        }
        theta['ratio'] = theta['sda'] / theta['r']
        true_theta_list.append(theta)
        datasets = hssm.simulate_data(model='shrink_spot', theta=theta, size=1000)
        sim_data_list.append(datasets)
    
    return true_theta_list, sim_data_list

def fit_model(i, dataset, true_theta, onnx_path):
    """fits the generated data to HSSM """
    theta_val = true_theta[i]
    ssm_model = hssm.HSSM(
        data=dataset,
        model="shrink_spot",
        model_config={
            "list_params": ["a", "z", "t", "target", "outer", "inner", "r", "sda"],
            "bounds": {
                "a": (0.3, 3.0),
                "z": (0.1, 0.9),
                "t": (0.001, 2),
                "target": (2.0, 5.5),
                "outer": (-5.5, 5.5),
                "inner": (-5.5, 5.5),
                "r": (0.01, 0.05),
                "sda": (1, 3),
            },
            "backend": "jax",
        },
        z=theta_val['z'],
        inner=theta_val['p_inner'],
        outer=theta_val['p_outer'],
        loglik=onnx_path,
        loglik_kind="approx_differentiable",
        p_outlier=0,
        include=[
            {"name": "r", "prior": {"name": "Uniform", "lower": 0.01, "upper": 0.05, 'initval': theta_val['r']}},
            {"name": "t", "prior": {"name": "Uniform", "lower": 0.001, "upper": 2, 'initval': theta_val['t']}},
            {"name": "sda", "prior": {"name": "Uniform", "lower": 1, "upper": 3, 'initval': theta_val['sda']}},
            {"name": "target", "prior": {"name": "Uniform", "lower": 2, "upper": 5.5, 'initval': theta_val['p_target']}},
        ]
    )
    return ssm_model

def infer_models(models, true_theta_list):
    """Performs posterior inference on the fitted model """
    results = []

    for i, model in enumerate(models):
        infer_ssp = model.sample(
            sampler="nuts_numpyro",
            cores=1,
            chains=4,
            draws=2000,
            tune=2500,
        )
        df = az.summary(infer_ssp)
        sda_val = infer_ssp.posterior['sda']
        r_val = infer_ssp.posterior['r']
        mean_ratio_val = np.mean(sda_val / r_val)

        result = {
            'true_theta': true_theta_list[i],
            'est_target': df.loc['target', 'mean'],
            'est_t': df.loc['t', 'mean'],
            'est_r': df.loc['r', 'mean'],
            'est_sda': df.loc['sda', 'mean'],
            'est_a': df.loc['a', 'mean'],
            'est_ratio': mean_ratio_val
        }
        results.append(result)
    
    return results

def save_results_to_csv(results, run_index):
    """Saves the results to a csv file"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/users/azhan378/hssm_estimates/results_{run_index}.csv', index=False)
    print(f"Results saved to results_{run_index}.csv")

def main():
    args = parse_arguments()
    run_index = args.run_index
    
    onnx_path = "/users/azhan378/LAN_pipeline_minimal/data/networks/torch/lan/shrink_spot/shrink_spot_lan_d38a4ee266ff11ef995fa0423f39a3e6_torch_model.onnx" #path where onnx file is saved 

    # Generates data
    true_theta_list, sim_data_list = generate_simulation_data(n_samples=4)

    # Fit models
    models = [fit_model(i, data, true_theta_list, onnx_path) for i, data in enumerate(sim_data_list)]

    # Infer results
    results = infer_models(models, true_theta_list)

    # Save results
    save_results_to_csv(results, run_index)

if __name__ == "__main__":
    main()
