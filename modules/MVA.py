import numpy as np

def mva_exato(N: int, S: list[float], Z: float = 0.0, print_table=True):
    """
    Calculates the performance of a closed queuing network using 
    Exact MVA for a single customer class.

    Args:
        N (int): The total number of customers (jobs) in the network.
        S (list): A list of service times (S_k) for each k-station.
                The length of the list is the number of stations (K).
        Z (float): The "think time" of the customers. The default is 0.

    Returns:
        A dictionary containing the final results (for N customers):
            - 'X': System throughput
            - 'R': List of residence/response times (R_k) for each station
            - 'Q': List of average queue sizes (Q_k) for each station
            - 'U': List of utilization (U_k) for each station
    """
    # Histórico para os gráficos
    history_X = []
    history_R_system = []

    K = len(S) 
    
    # Inicialização (n=0)
    N_k = np.zeros(K) # N_k[k] armazena N_k(n-1) no início de cada iteração
    R_k = np.zeros(K) # R_k armazena os tempos de residência R_k(n)
    X = 0.0
    R_system = 0.0
    
    # --- Impressão do Cabeçalho ---
    if print_table:
        print(f"\n--- Tabela MVA Exato (N={N}, Z={Z}) ---")
        headers_R = " | ".join([f"R_{k+1:<5}" for k in range(K)])
        headers_N = " | ".join([f"N_{k+1:<5}" for k in range(K)])
        print(f" n |   X(n)  | R_total | {headers_R} | {headers_N}")
        separator_line = "-" * (22 + (K * 9) + (K * 9))
        print(separator_line)
    # -------------------------------

    for n in range(1, N + 1):
        # 1. Calcular Tempos de Residência R_k(n)
        #    R_k(n) = S_k * (1 + N_k(n-1))
        for k in range(K):
            R_k[k] = S[k] * (1.0 + N_k[k])
        
        # 2. Calcular Throughput do Sistema X(n)
        R_total = np.sum(R_k)
        if (Z + R_total) == 0:
            X = 0.0
            R_system = 0.0
        else:
            X = n / (Z + R_total)
            R_system = (n / X) - Z
            
        # 3. Calcular Números da Fila N_k(n)
        #    N_k(n) = X(n) * R_k(n)
        for k in range(K):
            N_k[k] = X * R_k[k]

        # --- Armazenar Histórico ---
        history_X.append(X)
        history_R_system.append(R_system)
            
        # --- Imprimir Linha da Tabela ---
        if print_table:
            r_values = " | ".join([f"{r:<7.4f}" for r in R_k])
            n_values = " | ".join([f"{n_k:<7.4f}" for n_k in N_k])
            print(f"{n:<2} | {X:<7.4f} | {R_system:<7.4f} | {r_values} | {n_values}")
        # -------------------------------

    # 4. Calcular Utilização (U_k) (Passo final)
    U = [X * S[k] for k in range(K)]
    
    if print_table:
        print("-" * len(separator_line))
    
    return {
        'X': X,                                 # Throughtput X(N)
        'R_k': R_k,                             # Tempos de resposta R_k(N)
        'N_k': N_k,                             # Tamanhos de fila N_k(N)
        'U_k': U,                               # Utilização
        'R_total': R_system,                    # Tempo de resposta total R(N)
        'history_X': history_X,                 # Histórico do Throughput
        'history_R_total': history_R_system     # Histórico do tempo de resposta
    }


def mva_aproximado_bard(N: int, M: int, S: list[float], Z: float = 0.0, 
                         max_iter: int = 1000, tol: float = 1e-6, 
                         print_table: bool = True):
    """
    Calculates the approximate MVA (Mean Value Analysis) 
    using the Bard-Schweitzer algorithm (fixed-point iteration).

    Arguments:
        - N (int): The total number of clients (jobs) in the network.
        - M (int): The number of service centers (e.g., CPU, Disk).
        - S (list[float]): A list of average service times [S_1, S_2, ..., S_M].
        - Z (float): The average "think time". Default is 0.0.
        - max_iter (int): Maximum number of iterations for convergence.
        - tol (float): Tolerance for the stopping criterion.

    Returns:
        - dict: A dictionary containing the performance metrics for N clients.
    """

    if M != len(S):
        raise ValueError("[ERROR] The 'S' list of service times must have 'M' elements.")
    if N == 0:
        return {'X': 0.0, 'R_k': np.zeros(M), 'N_k': np.zeros(M), 'R_total': 0.0}

    history_X = []
    history_R_system = []
    
    # 1. Inicialização
    N_k = np.full(M, N / M)
    R_k = np.zeros(M)
    prop_N = (N - 1) / N
    X = 0.0
    R_system = 0.0

    # --- Impressão do Cabeçalho ---
    if print_table:
        print(f"\n--- Tabela MVA Aproximado (N={N}, Z={Z}, tol={tol}) ---")
        headers_R = " | ".join([f"R_{k+1:<5}" for k in range(M)])
        headers_N = " | ".join([f"N_{k+1:<5}" for k in range(M)])
        print(f"Iter |   X     | R_total | {headers_R} | {headers_N}")
        separator_line = "-" * (22 + (M * 9) + (M * 9))
        print(separator_line)
        # r_values_init = " | ".join(["-      " for _ in R_k])
        # n_values_init = " | ".join([f"{n_k:<7.4f}" for n_k in N_k])
        # print(f" 0   | -       | -       | {r_values_init} | {n_values_init}")
    # -------------------------------

    # 2. Iteração de Ponto Fixo
    for i in range(1, max_iter + 1):
        N_k_old = np.copy(N_k)
        
        # a. Calcular R_k
        for k in range(M):
            R_k[k] = S[k] * (1 + prop_N * N_k[k])
            
        # b. Calcular X
        R_total = np.sum(R_k)
        if (Z + R_total) == 0:
            X = 0.0
            R_system = 0.0
        else:
            X = N / (Z + R_total)
            R_system = (N / X) - Z
            
        # c. Calcular N_k
        for k in range(M):
            N_k[k] = X * R_k[k]

        # --- Armazenar Histórico ---
        history_X.append(X)
        history_R_system.append(R_system)
            
        # --- Imprimir Linha da Tabela ---
        if print_table:
            r_values = " | ".join([f"{r:<7.4f}" for r in R_k])
            n_values = " | ".join([f"{n_k:<7.4f}" for n_k in N_k])
            # Adiciona R_total ao print
            print(f"{i:<4} | {X:<7.4f} | {R_system:<7.4f} | {r_values} | {n_values}")
        # -------------------------------
            
        # 3. Verificar Convergência
        if np.max(np.abs(N_k - N_k_old)) < tol:
            break
    else:
        if print_table:
            print(f"[WARNING] The Approximate MVA didn't converge after {max_iter} iterations.")

    if print_table:
        print("-" * len(separator_line))
    
    return {
        'X': X,                                 # Throughtput X(N)
        'R_k': R_k,                             #  Tempos de resposta R_k(N)
        'N_k': N_k,                             # Tamanhos de fila N_k(N)
        'R_total': R_system,                    # Tempo de residência do sistema
        'history_X': history_X,                 # Histórico throughput
        'history_R_total': history_R_system     # Histórico tempo de resposta
    }