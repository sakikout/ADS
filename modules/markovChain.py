import pandas as pd
import numpy as np

def calculate_system_performance_markov(state_space: list[tuple[int, int, int]],
                                        max_r: int,
                                        cpu_rate: float, 
                                        fast_rate: float, 
                                        slow_rate: float):
    to_fast = cpu_rate * 0.5
    to_slow = cpu_rate * 0.5

    n = len(state_space)

    M = np.zeros((n, n))
    b = np.zeros(n)

    def idx_of(state):
        try:
            return state_space.index(state)
        except ValueError:
            return None

    for i, (c, f, s) in enumerate(state_space):
        
        total_out = 0
        if c > 0: total_out += cpu_rate
        if f > 0: total_out += fast_rate
        if s > 0: total_out += slow_rate
        
        M[i, i] = total_out

        if c < max_r and f > 0:
            j = idx_of((c+1, f-1, s))
            if j is not None:
                M[i, j] -= to_fast 

        if c < max_r and s > 0:
            j = idx_of((c+1, f, s-1))
            if j is not None:
                M[i, j] -= to_slow

        if c > 0 and f < max_r:
            j = idx_of((c-1, f+1, s))
            if j is not None:
                M[i, j] -= fast_rate 

        if c > 0 and s < max_r:
            j = idx_of((c-1, f, s+1))
            if j is not None:
                M[i, j] -= slow_rate

    M[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(M, b)

    cpu_busy_prob = sum(prob for prob, (c,_,_) in zip(pi, state_space) if c > 0)

    return {
        "u_cpu": cpu_busy_prob,
        "probabilities": list(zip(state_space, pi))
    }


def calculate_metrics_birth_death(max_capacity: int, 
                                  func_arrival, 
                                  func_service):

    status = np.arange(max_capacity + 1)
    df_status = pd.DataFrame({'k': status})

    df_status['arrival_rate'] = df_status['k'].apply(func_arrival)
    df_status['service_rate'] = df_status['k'].apply(func_service)

    lambdas = df_status['arrival_rate'].values
    mus = df_status['service_rate'].values
    
    coeficientes = np.zeros(len(status))
    coeficientes[0] = 1.0

    for k in range(1, len(status)):
        taxa_anterior = lambdas[k-1]
        taxa_atual_servico = mus[k]
        
        if taxa_atual_servico > 0:
            fator = taxa_anterior / taxa_atual_servico
            coeficientes[k] = coeficientes[k-1] * fator
        else:
            coeficientes[k] = 0.0

    df_status['coefficient'] = coeficientes

    soma_coeficientes = df_status['coefficient'].sum()
    p0 = 1.0 / soma_coeficientes
    
    df_status['probability'] = df_status['coefficient'] * p0

    num_medio_sistema_L = (df_status['k'] * df_status['probability']).sum()

    vazao_media_X = (df_status['service_rate'] * df_status['probability']).sum()
    
    if vazao_media_X > 0:
        tempo_medio_resposta_W = num_medio_sistema_L / vazao_media_X
    else:
        tempo_medio_resposta_W = 0.0

    prob_bloqueio = df_status.iloc[-1]['probability']

    return {
        "summary_metrics": {
            "L_avg_system_num": num_medio_sistema_L,
            "X_throughput": vazao_media_X,
            "W_avg_response_time": tempo_medio_resposta_W,
            "P_block": prob_bloqueio
        },
        "status_table": df_status[['k', 'arrival_rate', 'service_rate', 'probability']]
    }