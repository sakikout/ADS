import numpy as np
import math

def resolve_mm1_model(lam, mu, custo_servidor=0, custo_espera=0, nome="Sistema"):
    rho = lam / mu
    
    if rho >= 1:
        print(f"[{nome}] O sistema é INSTÁVEL (rho = {rho:.2f} >= 1)")
        return

    # Métricas M/M/1
    L = rho / (1 - rho)       # Número médio no sistema
    W = 1 / (mu - lam)        # Tempo médio no sistema
    Lq = rho**2 / (1 - rho)   # Número médio na fila
    Wq = rho / (mu - lam)     # Tempo médio na fila
    custo_total = custo_servidor + (L * custo_espera)

    print(f"--- Resultados: {nome} ---")
    print(f"Utilização (rho): {rho:.2%}")
    print(f"Nº médio no sistema (L): {L:.4f} clientes/máquinas")
    print(f"Tempo médio no sistema (W): {W:.4f} horas")
    print(f"Número médio na fila (Lq): {Lq:.4f} horas")
    print(f"Tempo médio na fila (Wq): {Wq:.4f} horas")

    if custo_espera > 0 or custo_servidor > 0:
        print(f"Custo Total Estimado: ${custo_total:.2f}/hora")
    print("-" * 30)

def resolve_jackson_network(taxas_externas, taxas_servico, matriz_roteamento):
    num_nos = len(taxas_externas)
    I = np.identity(num_nos)
    P_T = np.array(matriz_roteamento).T
    A = I - P_T
    gamma = np.array(taxas_externas)
    
    try:
        lambdas_efetivos = np.linalg.solve(A, gamma)

    except np.linalg.LinAlgError:
        print("Erro: Sistema impossível de resolver (verifique a matriz P)")
        return

    print("--- Resultados Rede de Jackson ---")
    L_total_sistema = 0
    
    for i in range(num_nos):
        lam = lambdas_efetivos[i]
        mu = taxas_servico[i]
        rho = lam / mu
        
        if rho >= 1:
            print(f"Estação {i+1} INSTÁVEL (rho={rho:.2f})")
            continue
            
        L_i = rho / (1 - rho) # M/M/1
        L_total_sistema += L_i
        
        print(f"Estação {i+1}: Lambda={lam:.2f}, Mu={mu}, Rho={rho:.2f}, L={L_i:.4f}")

    lambda_total_externo = sum(taxas_externas)
    W_sistema = L_total_sistema / lambda_total_externo
    
    print(f"\nTotal Clientes no Sistema (L): {L_total_sistema:.4f}")
    print(f"Tempo Médio no Sistema (W): {W_sistema:.4f} horas")

def erlang_c_prob(s, rho_total):
    if rho_total >= s: return 1.0
    
    numerador = (rho_total**s / math.factorial(s)) * (s / (s - rho_total))
    
    denominador = 0
    for k in range(s):
        denominador += (rho_total**k / math.factorial(k))
    denominador += numerador
    
    return numerador / denominador

def analize_mms_model(lam, mu, num_servidores):
    s = num_servidores
    rho_por_servidor = lam / (s * mu) # Utilização individual
    rho_total = lam / mu              # Intensidade de tráfego (Erlangs)

    if rho_por_servidor >= 1:
        return None # Instável

    prob_espera = erlang_c_prob(s, rho_total)
    Wq = prob_espera / (s*mu - lam)
    Wq_condicional = 1 / ((s * mu) - lam) 
    
    def prob_esperar_mais_que(tempo):
        return prob_espera * math.exp(-mu * (s - rho_total) * tempo)

    return {
        "s": s,
        "rho": rho_por_servidor,
        "P(Espera)": prob_espera,
        "P(W > 1min)": prob_esperar_mais_que(1),
        "Wq": Wq,
        "Wq_condicional": Wq_condicional
    }

def iterative_erlang_b(intensidade_trafego, num_canais):
    B = 1.0
    for s in range(1, num_canais + 1):
        B = (intensidade_trafego * B) / (s + intensidade_trafego * B)
    return B


# --------------------------- QUEUE STATS ------------------------------
def completed_mm1(lam, mu, k_prob=None, t_prob=None):
    """
    Análise completa M/M/1.
    Args:
        lam: Taxa de chegada
        mu: Taxa de serviço
        k_prob: (Opcional) Calcula Prob(N >= k)
        t_prob: (Opcional) Calcula Prob(TempoEspera > t)
    """
    rho = lam / mu
    
    print(f"\n=== ANÁLISE M/M/1 (Lambda={lam}, Mu={mu}) ===")
    
    if rho >= 1:
        print(f"ERRO: Sistema Instável (rho={rho:.2f} >= 1)")
        return None


    L = rho / (1 - rho)             # Nº médio no sistema
    Lq = rho**2 / (1 - rho)         # Nº médio na fila
    W = 1 / (mu - lam)              # Tempo médio no sistema
    Wq = rho / (mu - lam)           # Tempo médio na fila
    Var_L = rho / ((1 - rho)**2)    # Variância do número no sistema
    P0 = 1 - rho                    # Probabilidade do sistema vazio
    
    print(f"--- Médias ---")
    print(f"Utilização (rho):      {rho:.2%}")
    print(f"Nº no Sistema (L):     {L:.4f}")
    print(f"Nº na Fila (Lq):       {Lq:.4f}")
    print(f"Tempo no Sistema (W):  {W:.4f}")
    print(f"Tempo na Fila (Wq):    {Wq:.4f}")
    print(f"Prob. Vazio (P0):      {P0:.2%}")
    print(f"Variância do número no sistema:      {Var_L:.2%}")
    
    extra_metrics = {}
    
    if k_prob is not None:
        # P(N >= k) = rho^k
        p_ge_k = rho**k_prob
        print(f"Prob(N >= {k_prob}):        {p_ge_k:.2%}")
        extra_metrics['P(N>=k)'] = p_ge_k

    if t_prob is not None:
        # P(W > t) = e^(-mu(1-rho)t)
        p_wait_t = math.exp(-mu * (1 - rho) * t_prob)
        print(f"Prob(TempoSist > {t_prob}): {p_wait_t:.2%}")
        extra_metrics['P(W>t)'] = p_wait_t

    return {"L": L, "W": W, "Lq": Lq, "Wq": Wq, "rho": rho, "P0": P0, **extra_metrics}

def completed_mms(lam, mu, s, t_target=None):
    """
    Análise completa M/M/s (Erlang-C).
    Args:
        s: Número de servidores
        t_target: (Opcional) Calcula Prob(Espera na fila > t)
    """
    rho_servidor = lam / (s * mu) # Utilização per capita
    intensidade_A = lam / mu      # Tráfego em Erlangs
    
    print(f"\n=== ANÁLISE M/M/{s} (Lambda={lam}, Mu={mu}) ===")
    
    if rho_servidor >= 1:
        print(f"ERRO: Sistema Instável (rho={rho_servidor:.2f} >= 1)")
        return None

    # Probabilidade de zero clientes no sistema
    sum_part = sum([(intensidade_A**n)/math.factorial(n) for n in range(s)])
    term_s = (intensidade_A**s) / math.factorial(s)
    term_rho = 1 / (1 - rho_servidor)
    P0 = 1 / (sum_part + (term_s * term_rho))

    # Probabilidade de Espera (Erlang-C / P(Wait))
    P_wait = (term_s * term_rho) * P0                   # Probabilidade de chegar e encontrar todos ocupados
    
    Lq = (P_wait * rho_servidor) / (1 - rho_servidor)   # Média na fila
    L = Lq + intensidade_A                              # Média no sistema
    Wq = Lq / lam                                       # Tempo na fila
    W = Wq + (1/mu)                                     # Tempo no sistema
    
    print(f"--- Performance ---")
    print(f"Utilização por servidor: {rho_servidor:.2%}")
    print(f"Prob. de Espera (Erlang-C): {P_wait:.2%} (Chance de pegar fila)")
    print(f"Prob. Sistema Vazio (P0):   {P0:.4f}")
    print(f"--- Médias ---")
    print(f"Nº na Fila (Lq):    {Lq:.4f}")
    print(f"Nº no Sistema (L):  {L:.4f}")
    print(f"Tempo Fila (Wq):    {Wq:.4f}")
    print(f"Tempo Sist (W):     {W:.4f}")
    
    if t_target is not None:
        # P(Wq > t) = P(Wait) * e^(-mu(s - A)t)
        prob_esperar_mais_t = P_wait * math.exp(-mu * (s - intensidade_A) * t_target)
        print(f"--- Distribuição ---")
        print(f"Prob(Espera > {t_target}): {prob_esperar_mais_t:.2%}")

    return {"rho": rho_servidor, "P_wait": P_wait, "Lq": Lq, "Wq": Wq}

def completed_jackson_network(taxas_ext, taxas_servico, matriz_roteamento):
    """
    Resolve Rede de Jackson Aberta.
    Args:
        taxas_ext: vetor gamma (chegadas externas)
        taxas_servico: vetor mu
        matriz_roteamento: Matriz de probabilidades P[i][j]
    """
    num_nos = len(taxas_ext)
    gamma = np.array(taxas_ext)
    P = np.array(matriz_roteamento)
    
    print(f"\n=== ANÁLISE DE REDE DE JACKSON ({num_nos} Nós) ===")
    I = np.identity(num_nos)
    try:
        lambdas = np.linalg.solve(I - P.T, gamma)
    except:
        print("Erro: Matriz singular. Verifique as probabilidades.")
        return

    L_total = 0
    W_total_sistema = 0
    results = []
    
    print(f"{'Nó':<3} | {'Lambda':<8} | {'Mu':<6} | {'Rho':<8} | {'L (cli)':<8} | {'W (tempo)':<8}")
    print("-" * 65)

    gargalo_val = -1
    gargalo_id = -1

    for i in range(num_nos):
        lam = lambdas[i]
        mu = taxas_servico[i]
        rho = lam / mu

        if rho > gargalo_val:
            gargalo_val = rho
            gargalo_id = i + 1

        if rho >= 1:
            status = "INSTÁVEL"
            L_i, W_i = float('inf'), float('inf')
        else:
            L_i = rho / (1 - rho)
            W_i = 1 / (mu - lam)
            status = f"{rho:.2%}"
            L_total += L_i
            
        print(f"{i+1:<3} | {lam:<8.2f} | {mu:<6.2f} | {status:<8} | {L_i:<8.2f} | {W_i:<8.4f}")
        results.append({"id": i+1, "lambda": lam, "rho": rho, "L": L_i, "W": W_i})

    lambda_total_externo = sum(gamma)
    W_sistema = L_total / lambda_total_externo
    
    print("-" * 65)
    print(f"Gargalo do Sistema:   Nó {gargalo_id} (Utilização: {gargalo_val:.2%})")
    print(f"População Média (L):  {L_total:.4f} clientes")
    print(f"Tempo de Ciclo (W):   {W_sistema:.4f} unidades de tempo")
    
    return {"L_total": L_total, "W_sistema": W_sistema, "nos": results}

def advanced_erlang_b(lam, mu, s, meta_bloqueio=None):
    """
    Análise M/M/s/s (Perda).

    Args:
        lam: Taxa de chegada
        mu: Taxa de serviço
        s: Número de servidores
        meta_bloqueio: Se passado, ignora s atual e encontra s ideal.
    """
    intensidade_A = lam / mu
    print(f"\n=== ANÁLISE ERLANG-B (A={intensidade_A:.1f} Erlangs) ===")

    if meta_bloqueio is not None:
        s_teste = 1
        while True:
            b_teste = iterative_erlang_b(intensidade_A, s_teste)
            if b_teste <= meta_bloqueio:
                print(f"Meta: Bloqueio <= {meta_bloqueio:.2%}")
                print(f"Solução: Necessários {s_teste} canais (Bloqueio atual: {b_teste:.4%})")
                return s_teste
            s_teste += 1
            if s_teste > 10000: break
    
    prob_bloqueio = iterative_erlang_b(intensidade_A, s)
    # Métricas de Fluxo
    lambda_efetivo = lam * (1 - prob_bloqueio) # Quem realmente entra
    lambda_perdido = lam * prob_bloqueio       # Quem leva sinal de ocupado
    utilizacao_sistema = lambda_efetivo / (s * mu) # Quanto o sistema está ocupado
    
    print(f"Capacidade (s):        {s} canais")
    print(f"Prob. [cite_start]Bloqueio (Pb):   {prob_bloqueio:.4%} [cite: 36, 37]")
    print(f"--- Fluxo ---")
    print(f"Taxa Chegada:          {lam:.2f}/min")
    print(f"Taxa Efetiva (Entra):  {lambda_efetivo:.2f}/min")
    print(f"Taxa Perda (Rejeita):  {lambda_perdido:.2f}/min")
    print(f"Utilização Média:      {utilizacao_sistema:.2%}")
    
    return {"Pb": prob_bloqueio, "Eficiencia": 1-prob_bloqueio}