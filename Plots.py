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
    
def plot_V_cosmo(smiles, real, prediction):
    plt.figure(figsize=(4, 4))
    
    if not real:
        real = 0
        
    labels = ['Real', 'Prediction']
    V_cosmo = [real, prediction]

    bars = plt.bar(labels, V_cosmo, color=['dodgerblue', 'red'], width=0.8)

    plt.ylabel(r'$V_{COSMO}(\AA^3)$')
    plt.title(smiles)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    
    plt.show()