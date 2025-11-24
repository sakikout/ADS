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
