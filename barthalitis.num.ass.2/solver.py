#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def get_params(c=0.5, G=1.7, T=1.7, i=0.04):
    return {
        'C_bar': 0.6,
        'I_bar': 0.2,
        'c': c,
        'alpha': 0.1,
        'b': 0.1,
        'G': G,
        'T': T,
        'i': i
    }

def steady_state(params):
    C_bar = params['C_bar']
    I_bar = params['I_bar']
    c = params['c']
    alpha = params['alpha']
    b = params['b']
    G = params['G']
    T = params['T']
    i = params['i']
    D = C_bar + I_bar + G - c * T - b * i
    beta = c + alpha
    Y_ss = D / (1 - beta)
    C_ss = C_bar + c * (Y_ss - T)
    I_ss = I_bar + alpha * Y_ss - b * i
    return Y_ss, C_ss, I_ss, D, beta

def dynamic_path(Y_init, params, periods=20):
    C_bar = params['C_bar']
    I_bar = params['I_bar']
    c = params['c']
    alpha = params['alpha']
    b = params['b']
    T = params['T']
    i = params['i']
    _, _, _, D, beta = steady_state(params)
    Y = np.zeros(periods + 1)
    C = np.zeros(periods + 1)
    I = np.zeros(periods + 1)
    Y[0] = D + beta * Y_init  # t=0 uses Y_{-1}
    C[0] = C_bar + c * (Y_init - T)
    I[0] = I_bar + alpha * Y_init - b * i
    for t in range(1, periods + 1):
        Y[t] = D + beta * Y[t-1]
        C[t] = C_bar + c * (Y[t-1] - T)
        I[t] = I_bar + alpha * Y[t-1] - b * i
    return Y, C, I

# Exercise 1: Base steady state (c=0.5)
base_params = get_params(c=0.5)
Y_ss, C_ss, I_ss, D, beta = steady_state(base_params)
print("Exercise 1: Steady State for base parameters (c=0.5)")
print(f"Y_ss = {Y_ss:.4f}, C_ss = {C_ss:.4f}, I_ss = {I_ss:.4f}")

# Exercise 2: Dynamic transition from Y_{-1} = 0.9 Y_ss
c_values = [0.4, 0.5, 0.6]
periods = 20
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
Y_paths = {}
for c_val in c_values:
    p = get_params(c=c_val)
    Y_ss_i, _, _, _, _ = steady_state(p)
    Y0 = 0.9 * Y_ss_i
    Ydyn, Cdyn, Idyn = dynamic_path(Y0, p, periods)
    t = np.arange(0, periods+1)
    Y_paths[c_val] = Ydyn
    axs[0].plot(t, Ydyn, marker='o', label=f'c = {c_val}')
    axs[1].plot(t, Cdyn, marker='s', label=f'c = {c_val}')
    axs[2].plot(t, Idyn, marker='^', label=f'c = {c_val}')
axs[0].set_title('Dynamic Path: Output (Y)')
axs[1].set_title('Dynamic Path: Consumption (C)')
axs[2].set_title('Dynamic Path: Investment (I)')
for ax in axs:
    ax.legend()
    ax.grid(True)
axs[2].set_xlabel('Periods')
plt.tight_layout()
plt.show()

# Exercise 2 (continued): Overlay Y paths for different c
plt.figure(figsize=(8,5))
for c_val, Ydyn in Y_paths.items():
    plt.plot(np.arange(0, periods+1), Ydyn, marker='o', label=f'c = {c_val}')
plt.title('Dynamic Transition of Y for different c')
plt.xlabel('Periods')
plt.ylabel('Output Y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Exercise 3: Policy Changes (base c=0.5)
gamma = 0.7
policy_scenarios = {
    'Gov Spending +50%': {'G': 1.7 * 1.5, 'T': 1.7, 'i': 0.04},
    'Taxes +30%': {'G': 1.7, 'T': 1.7 * 1.3, 'i': 0.04},
    'Interest Rate +100bps': {'G': 1.7, 'T': 1.7, 'i': 0.04 + 0.01}
}

for name, change in policy_scenarios.items():
    p_new = get_params(c=0.5, G=change['G'], T=change['T'], i=change['i'])
    Y_ss_new, C_ss_new, I_ss_new, _, _ = steady_state(p_new)
    # initial condition is the base steady state (from Exercise 1)
    Y_init = Y_ss  # starting at original steady state
    Y_dyn, C_dyn, I_dyn = dynamic_path(Y_init, p_new, periods)
    L_dyn = gamma * Y_dyn
    t = np.arange(0, periods+1)
    
    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    axs[0,0].plot(t, Y_dyn, 'o-')
    axs[0,0].axhline(Y_ss, color='grey', linestyle='--', label='Old SS')
    axs[0,0].axhline(Y_ss_new, color='red', linestyle='--', label='New SS')
    axs[0,0].set_title(f"Output (Y) - {name}")
    axs[0,0].legend(); axs[0,0].grid(True)
    
    axs[0,1].plot(t, C_dyn, 's-')
    axs[0,1].set_title("Consumption (C)"); axs[0,1].grid(True)
    
    axs[1,0].plot(t, I_dyn, '^-')
    axs[1,0].set_title("Investment (I)"); axs[1,0].grid(True)
    
    axs[1,1].plot(t, L_dyn, 'd-')
    axs[1,1].set_title("Employment (L)"); axs[1,1].grid(True)
    
    plt.suptitle(f"Policy Change: {name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()