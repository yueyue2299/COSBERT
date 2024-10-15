import numpy as np
import matplotlib.pyplot as plt

def plot_sigma_profile(sigma_profile, prediction, smiles, id=None):
    density = np.round(np.arange(-0.025, 0.0251, 0.001), 4)

    plt.figure(figsize=(8, 6))
    if sigma_profile:
        plt.plot(density, sigma_profile, marker='o', linestyle='-', color='dodgerblue', markersize=5, label='Real Profile')
    plt.plot(density, prediction, marker='s', linestyle='--', color='r', markersize=5, label='Prediction')

    plt.xlabel(r'$\sigma(e/\AA^2)$')
    plt.ylabel(r'P($\sigma$)')
    if id:
        plt.title(f'{id}, {smiles}')
    else:
        plt.title(smiles)
    plt.grid(True)
    plt.legend()

    plt.show()