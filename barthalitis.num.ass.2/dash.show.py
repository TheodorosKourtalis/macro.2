#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:17:22 2025

@author: thodoreskourtales
"""

#!/usr/bin/env python
import numpy as np
from dash import Dash, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###############################################################################
#                             Economic Model Functions                        #
###############################################################################
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
    _, _, _, D, beta = steady_state(params)
    C_bar = params['C_bar']
    I_bar = params['I_bar']
    c = params['c']
    alpha = params['alpha']
    b = params['b']
    T = params['T']
    i = params['i']
    
    Y = np.zeros(periods + 1)
    C = np.zeros(periods + 1)
    I = np.zeros(periods + 1)
    
    # Use Y_{-1} = Y_init for t = 0
    Y[0] = D + beta * Y_init
    C[0] = C_bar + c * (Y_init - T)
    I[0] = I_bar + alpha * Y_init - b * i
    
    for t in range(1, periods + 1):
        Y[t] = D + beta * Y[t-1]
        C[t] = C_bar + c * (Y[t-1] - T)
        I[t] = I_bar + alpha * Y[t-1] - b * i
        
    return Y, C, I

###############################################################################
#                        Precompute Data and Build Figures                    #
###############################################################################
# Base steady state using c = 0.5
base_params = get_params(c=0.5)
Y_ss, C_ss, I_ss, D, beta = steady_state(base_params)

# Steady state summary: use inline math with dollar signs
steady_state_text = """
**Steady-State (Base: c = 0.5)**

- **Output:** $Y_{ss} = 4.1150$
- **Consumption:** $C_{ss} = 1.8075$
- **Investment:** $I_{ss} = 0.6075$
"""

# Exercise 2: Dynamic transitions for different c values
c_values = [0.4, 0.5, 0.6]
periods = 20
t_vals = np.arange(periods + 1)

# Create a 3-row subplot figure for Y, C and I paths
fig_dynamic = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        "Output $Y_t$", 
        "Consumption $C_t$", 
        "Investment $I_t$"
    ),
    shared_xaxes=True
)

for c_val in c_values:
    p = get_params(c=c_val)
    Y_ss_i, _, _, _, _ = steady_state(p)
    Y0 = 0.9 * Y_ss_i  # start at 90% of the steady state for this c
    Ydyn, Cdyn, Idyn = dynamic_path(Y0, p, periods)
    
    fig_dynamic.add_trace(
        go.Scatter(x=t_vals, y=Ydyn, mode='lines+markers', name=f'c = {c_val}'),
        row=1, col=1
    )
    fig_dynamic.add_trace(
        go.Scatter(x=t_vals, y=Cdyn, mode='lines+markers', name=f'c = {c_val}', showlegend=False),
        row=2, col=1
    )
    fig_dynamic.add_trace(
        go.Scatter(x=t_vals, y=Idyn, mode='lines+markers', name=f'c = {c_val}', showlegend=False),
        row=3, col=1
    )

fig_dynamic.update_layout(
    title_text="Dynamic Paths for $Y_t$, $C_t$, and $I_t$ for different $c$ values",
    height=700
)

# Create an overlay figure for just the Y paths
fig_overlay = go.Figure()
for c_val in c_values:
    p = get_params(c=c_val)
    Y_ss_i, _, _, _, _ = steady_state(p)
    Y0 = 0.9 * Y_ss_i
    Ydyn, _, _ = dynamic_path(Y0, p, periods)
    fig_overlay.add_trace(
        go.Scatter(x=t_vals, y=Ydyn, mode='lines+markers', name=f'c = {c_val}')
    )

fig_overlay.update_layout(
    title="Overlay: Dynamic Transition of $Y_t$ for different $c$",
    xaxis_title="Periods",
    yaxis_title="Output $Y_t$"
)

# Exercise 3: Policy scenarios for c = 0.5
gamma = 0.7
policy_scenarios = {
    "Gov Spending +50%": {"G": 1.7 * 1.5, "T": 1.7, "i": 0.04},
    "Taxes +30%": {"G": 1.7, "T": 1.7 * 1.3, "i": 0.04},
    "Interest Rate +100bps": {"G": 1.7, "T": 1.7, "i": 0.04 + 0.01}
}

policy_figures = {}
for scenario, changes in policy_scenarios.items():
    p_new = get_params(c=0.5, G=changes["G"], T=changes["T"], i=changes["i"])
    Y_ss_new, _, _, _, _ = steady_state(p_new)
    
    # Use the base steady state as the starting point
    Y_init = Y_ss
    Y_dyn, C_dyn, I_dyn = dynamic_path(Y_init, p_new, periods)
    L_dyn = gamma * Y_dyn  # Employment = gamma * Y
    
    fig_policy = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Output $Y_t$", 
            "Consumption $C_t$",
            "Investment $I_t$", 
            "Employment $L_t = \\gamma Y_t$"
        )
    )
    # Add Output trace with old and new steady state lines
    fig_policy.add_trace(
        go.Scatter(x=t_vals, y=Y_dyn, mode='lines+markers', name="Output"),
        row=1, col=1
    )
    fig_policy.add_trace(
        go.Scatter(
            x=[t_vals[0], t_vals[-1]], 
            y=[Y_ss, Y_ss],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name="Old SS"
        ),
        row=1, col=1
    )
    fig_policy.add_trace(
        go.Scatter(
            x=[t_vals[0], t_vals[-1]], 
            y=[Y_ss_new, Y_ss_new],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name="New SS"
        ),
        row=1, col=1
    )
    # Add Consumption, Investment and Employment traces
    fig_policy.add_trace(
        go.Scatter(x=t_vals, y=C_dyn, mode='lines+markers', name="Consumption"),
        row=1, col=2
    )
    fig_policy.add_trace(
        go.Scatter(x=t_vals, y=I_dyn, mode='lines+markers', name="Investment"),
        row=2, col=1
    )
    fig_policy.add_trace(
        go.Scatter(x=t_vals, y=L_dyn, mode='lines+markers', name="Employment"),
        row=2, col=2
    )
    fig_policy.update_layout(title_text=f"Policy Change: {scenario}", height=600)
    policy_figures[scenario] = fig_policy

###############################################################################
#                          Dash App Layout with Tailwind                      #
###############################################################################
# Use up-to-date external scripts (MathJax 2.7.9) and Tailwind CSS from CDN.
external_scripts = [
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-MML-AM_CHTML"
]
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
]

app = Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
app.title = "Economic Model Simulation"

# Model equations Markdown using $$ for display math
model_equations = r"""
## Model Equations

**Steady-State Equations:**

$$
D = \bar{C} + \bar{I} + G - c\,T - b\,i
$$

$$
\beta = c + \alpha
$$

$$
Y_{ss} = \frac{D}{1-\beta}
$$

$$
C_{ss} = \bar{C} + c\,(Y_{ss} - T)
$$

$$
I_{ss} = \bar{I} + \alpha\,Y_{ss} - b\,i
$$

**Dynamic Path Equations:**

$$
Y_t = D + \beta\,Y_{t-1}
$$

$$
C_t = \bar{C} + c\,(Y_{t-1} - T)
$$

$$
I_t = \bar{I} + \alpha\,Y_{t-1} - b\,i
$$
"""

app.layout = html.Div(className="bg-gray-100 min-h-screen", children=[
    # Header
    html.Div(className="container mx-auto px-4 py-8", children=[
        html.H1("Economic Model Simulation", 
                className="text-4xl font-bold text-center text-gray-800"),
        html.P("A dynamic simulation with modern design using Tailwind CSS and Plotly Dash.",
               className="text-center text-lg text-gray-600")
    ]),
    
    # Model Equations & Steady-State Summary
    html.Div(className="container mx-auto px-4 py-4", children=[
        html.Div(className="bg-white shadow-lg rounded-lg p-6 mb-8", children=[
            dcc.Markdown(model_equations, mathjax=True, className="prose max-w-none"),
            html.Div(
                dcc.Markdown(steady_state_text, mathjax=True),
                className="mt-6 text-lg text-gray-700"
            )
        ])
    ]),
    
    # Dynamic Transition Section
    html.Div(className="container mx-auto px-4 py-4", children=[
        html.H2("Dynamic Transition", className="text-2xl font-semibold text-gray-800 mb-4"),
        html.Div(className="bg-white shadow-lg rounded-lg p-6 mb-8", children=[
            dcc.Graph(figure=fig_dynamic)
        ]),
        html.Div(className="bg-white shadow-lg rounded-lg p-6 mb-8", children=[
            dcc.Graph(figure=fig_overlay)
        ])
    ]),
    
    # Policy Changes Section (each policy in its own container for clarity)
    html.Div(className="container mx-auto px-4 py-4", children=[
        html.H2("Policy Changes", className="text-2xl font-semibold text-gray-800 mb-4"),
        html.Div(className="mb-8", children=[
            html.Div(className="bg-white shadow-lg rounded-lg p-6 mb-6", children=[
                html.H3("Gov Spending +50%", className="text-xl font-bold text-gray-800 mb-2"),
                dcc.Graph(figure=policy_figures["Gov Spending +50%"])
            ]),
            html.Div(className="bg-white shadow-lg rounded-lg p-6 mb-6", children=[
                html.H3("Taxes +30%", className="text-xl font-bold text-gray-800 mb-2"),
                dcc.Graph(figure=policy_figures["Taxes +30%"])
            ]),
            html.Div(className="bg-white shadow-lg rounded-lg p-6 mb-6", children=[
                html.H3("Interest Rate +100bps", className="text-xl font-bold text-gray-800 mb-2"),
                dcc.Graph(figure=policy_figures["Interest Rate +100bps"])
            ])
        ])
    ]),
    
    # Footer
    html.Div(className="container mx-auto px-4 py-4", children=[
        html.P("© 2025 Economic Model Simulation", 
               className="text-center text-gray-500 text-sm")
    ])
])

###############################################################################
#                                   Run Server                                 #
###############################################################################
if __name__ == '__main__':
    app.run(debug=True)