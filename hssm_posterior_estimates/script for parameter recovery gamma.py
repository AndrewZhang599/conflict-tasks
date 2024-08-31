import argparse
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import hssm
import ssms
import pymc as pm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model fitting')
    parser.add_argument('--run_index', type=int, required=True, help='index')
    return parser.parse_args()

def generate_simulation_data(n_samples):
    """=generate data using randomly generated parameter values"""
    true_theta_list = []
    sim_data_list = []

    for _ in range(n_samples):
        theta = {
            'v': np.random.uniform(-3.0, 3.0),
            'a': np.random.uniform(0.3, 3.0),
            'z': np.random.uniform(0.1, 0.9),
            't': np.random.uniform(0.001, 2),
            'shape': np.random.uniform(2.0, 10),
            'scale': np.random.uniform(0.01, 1.0),
            'c': np.random.uniform(-3.0, 3.0),
        }
        true_theta_list.append(theta)
        dataset = hssm.simulate_data(model='gamma_drift', theta=theta, size=1000)
        sim_data_list.append(dataset)

    return true_theta_list, sim_data_list

def fit_model(dataset, onnx_path):
    """fit the HSSM model with the generated data"""
    gamma_model = hssm.HSSM(
        model='gamma_drift',
        model_config={
            "list_params": ["v", "a", "z", "t", "shape", "scale", "c"],
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 3.0),
                "z": (0.1, 0.9),
                "t": (0.001, 2),
                "shape": (2.0, 10),
                "scale": (0.01, 1.0),
                "c": (-3.0, 3.0)
            },
            "backend": "jax",
        },
        data=dataset,
        loglik=onnx_path,
        loglik_kind="approx_differentiable",
        p_outlier=0
    )
    return gamma_model

def infer_models(models, true_theta_list):
    """inference using HSSM on the simulated data"""
    results = []

    for i, model in enumerate(models):
        infer_ssp = model.sample(
            sampler="nuts_numpyro",
            cores=1,
            chains=4,
            draws=1500,
            tune=2000,
        )
        df = az.summary(infer_ssp)

        result = {
            'true_theta': true_theta_list[i],
            'est_v': df.loc['v', 'mean'],
            'est_a': df.loc['a', 'mean'],
            'est_z': df.loc['z', 'mean'],
            'est_t': df.loc['t', 'mean'],
            'est_shape': df.loc['shape', 'mean'],
            'est_scale': df.loc['scale', 'mean'],
            'est_c': df.loc['c', 'mean']
        }
        results.append(result)

    return results

def save_results_to_csv(results, run_index):
    """Save the results to a CSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/users/azhan378/hssm_estimates/gamma_generated_data/results_gamma_{run_index}.csv', index=False)
    print(f"Results saved to results_gamma_{run_index}.csv")

def main():
    args = parse_arguments()
    run_index = args.run_index
    
    onnx_path = "/users/azhan378/hssm_estimates/onnx modles/gamma_drift_lan_eeccd94c421811ef8aeaa0423f3e9b72_torch_model.onnx"

    # Generate simulation data
    true_theta_list, sim_data_list = generate_simulation_data(n_samples=4)

    # Fit models to the simulated data
    models = [fit_model(data, onnx_path) for data in sim_data_list]

    # Perform inference on the models
    results = infer_models(models, true_theta_list)

    # Save the results to a CSV file
    save_results_to_csv(results, run_index)

if __name__ == "__main__":
    main()
