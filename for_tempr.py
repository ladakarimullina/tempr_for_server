from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import random
from scipy.integrate import quad
from scipy.integrate import quad_vec
import os
from functools import lru_cache
import argparse
import ast
from collections import defaultdict
import re

# Constants
hbar = 1.054e-34  # Reduced Planck constant in J·s
m_e = 9.109e-31   # Electron mass in kg (note: corrected from 10^-51 to 10^-31)
J_to_eV = 6.242e18  # Conversion factor from Joules to eV
m_to_nm = 1e9      # Conversion factor from nanometers to meters


# Calculate the expression
numerator = hbar**2
denominator = 2 * m_e
result = (numerator / denominator) * J_to_eV * (m_to_nm**2)
DIM_LENGTH = result  # h^2 / (2m_e) * (joule to eV) * (m^2 to nm^2) [eV*nm^2]
class SingleChannelCurrent:
    e = 1
    k = 8.62e-5  # Boltzmann constant [eV/K]

    def __init__(self, eps_0=5., U_0=20., epsF=5.):
        self.U_0 = U_0
        self.eps_0 = eps_0
        self.epsF = epsF
        self.a_0 = np.sqrt((self.U_0 - self.eps_0)/DIM_LENGTH)  
        self.a_F = np.sqrt((self.U_0 - self.epsF)/DIM_LENGTH) 

    def set_m(self, m):
        self.m = m
        
    def d(self, eps, r, to_print=False):
        
        pi = np.pi

        a = np.sqrt((self.U_0 - eps) / DIM_LENGTH)
        k = np.sqrt(eps / DIM_LENGTH)

        # --- h ---
        h = np.exp(-a * r) / (4 * pi * r)
        h[0] = np.exp(-2 * a * r[0]) / (8 * pi * r[0])
        h[-1] = np.exp(-2 * a * r[-1]) / (8 * pi * r[-1])

        # --- inverse_mu_h ---
        if abs(eps - self.eps_0) < 1e-8:
            inverse_mu_h = np.zeros_like(h)
        else:
            mu = (4 * pi) / (self.a_0 - a)
            inverse_mu_h = 1 / (mu * h)

        m = self.m

        # --- iterative alpha / beta ---
        alpha = np.zeros(m + 1, dtype=complex)
        beta = np.zeros(m + 1, dtype=complex)

        z = (a + 1j * k) / (a - 1j * k)

        # base case (m = 2)
        aux = inverse_mu_h[0] + z
        alpha[2] = - (h[1] / h[0]) / aux
        beta[2] = 1 / aux

        # forward sweep
        for j in range(3, m + 1):
            aux_alpha = inverse_mu_h[j - 1]
            alpha[j] = -1 / (aux_alpha + (h[j - 2] / h[j - 1]) * alpha[j - 1])

            aux_beta = inverse_mu_h[j - 2]
            beta[j] = -beta[j - 1] / (aux_beta + alpha[j - 1])

        alpha_m = alpha[m]
        beta_m = beta[m]

        # --- final expression ---
        aux = inverse_mu_h[-1] + z
        h_coef = h[-2] / h[-1]

        progon = h_coef * beta_m / (aux + h_coef * alpha_m)

        nomin = a**2 * k**2 * np.exp(-2 * a * (r[-1] + r[0]))

        U0_eff = a**2 + k**2
        denom = 4 * U0_eff**2 * r[0] * r[-1] * pi**2 * h[0]**2

        ans = nomin / denom * np.abs(progon)**2

        return ans

    def nF(self, eps, T):
        arg = (eps - self.epsF) / (self.k * T)
        if arg > 13.8:  # np.log(1e6) = 13.8 - just some large value
            return np.exp(-arg)
        return 1 / (np.exp(arg) + 1)

    def i_m(self, r, V, T):
        def integrand_i_m(eps):
            G0 = 7.75e-5
            # return G0*(self.nF(eps - self.e * V / 2, T) - self.nF(eps + self.e * V / 2, T)) * self.d(eps, r)
            return G0*(self.nF(eps, T) - self.nF(eps + V, T)) * self.d(eps, r) 

        return quad_vec(integrand_i_m, a=0, b=self.U_0)[0]

def get_all_r_values(file_path):

    r_dict = {}
    id_counts = defaultdict(int)

    with open(file_path, encoding="utf-8-sig") as file:

        for line in file:

            line = line.strip()

            if line.startswith("r in chain"):

                try:
                    chain_id, list_str = line.split(":", 1)

                    chain_id = chain_id.replace("r in chain", "").strip()

                    r_list = ast.literal_eval(list_str.strip())
                    
                    # Count occurrences
                    count = id_counts[chain_id]
                    id_counts[chain_id] += 1

                    # Rename if duplicate
                    if count == 0:
                        final_id = chain_id
                    else:
                        final_id = f"{chain_id}_{count}"

                    r_dict[final_id] = r_list

                except Exception as e:
                    print("Failed:", line)
                    print(e)

    return r_dict


def compute_current(chain, r_array_list, v):
    # eps_0 = 3.25
    eps_0 = 3.1
    epsF = 3.1
    U_0 = 3.55
    
    current_instance = SingleChannelCurrent(eps_0=eps_0, U_0=U_0, epsF=epsF) 
    r_array = np.array(r_array_list)
    m = len(r_array) - 1
    current_instance.set_m(m)
    return 1e6 * current_instance.i_m(r_array, V=v, T=300)

########################################## MAIN PART ########################################## 

T = np.linspace(150, 280, 6)
T = np.append(T, 300)

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--data_path', type=str, required=True,
                    help='Base directory for input data files')
parser.add_argument('--mode', type=str, required=True,
                    help='V_set or V_set_neg or V_ or V_neg')
parser.add_argument('--v', type=int, required=True)

parser.add_argument('--cycle', type=int, required=True)

args = parser.parse_args()
base_path = args.data_path

v = 0.5
eps_0 = 3.1
epsF = 3.1
U_0 = 3.55


os.makedirs(base_path, exist_ok=True)
for T in T_array:
    with open(os.path.join(base_path, f'i_{int(T)}.txt'), 'w') as f:
        pass


for T in T_array:

    print(f"\n=== Temperature {T} K ===")

    # progress over cycles
    for cycle in tqdm(cycles, desc=f"T = {int(T)} K", position=0):

        file_path = os.path.join(
            base_path,
            f'V_set_7/020_015/r_in_chains_{cycle}.txt'
        )
        r_values_dict = get_all_r_values(file_path)

        # sort chains (important!)
        items = list(r_values_dict.items())
        items.sort(key=lambda x: len(x[1]), reverse=True)

        # --- worker ---
        def task(chain, r_array_list):
            current_instance = SingleChannelCurrent(
                eps_0=eps_0,
                U_0=U_0,
                epsF=epsF
            )
            r_array = np.array(r_array_list)
            m = len(r_array) - 1
            current_instance.set_m(m)
            return 1e6 * current_instance.i_m(r_array, V=v, T=T)

        # --- parallel over chains ---
        results = Parallel(
            n_jobs=-1,
            batch_size=1
        )(
            delayed(task)(chain, r_array_list)
            for chain, r_array_list in items
        )

        total_current = sum(results)

        # --- write result ---
        with open(os.path.join(base_path, f'i_{int(T)}.txt'), 'a') as f:
            f.write(f'{cycle}: {total_current}\n')
