import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_performance_graphs(mva_results: dict, 
                            D_k_list: list[float], Z:float, 
                            N_max: int, title_suffix: str):
    mva_X = mva_results['history_X']
    mva_R = mva_results['history_R_total']
    n_clientes = list(range(1, N_max + 1))
    D_total = np.sum(D_k_list)
    D_max = np.max(D_k_list)
    idx_gargalo = np.argmax(D_k_list)

    N_star = (D_total + Z) / D_max

    print(f"D_total (D): {D_total:.3f} s")
    print(f"D_max (Gargalo - CPU): {D_max:.3f} s (Recurso {idx_gargalo+1})")
    print(f"Tempo de Pensar (Z): {Z:.1f} s")
    print(f"Ponto de Saturação (N*): (D+Z)/D_max = {N_star:.2f}")

    # X_otimista = min(1/D_max, n / (D_total + Z))
    X_lim_otimista = [min(1/D_max, n / (D_total + Z)) for n in n_clientes]
    # X_pessimista = n / (n*D_total + Z)
    X_lim_pessimista = [n / (n * D_total + Z) for n in n_clientes]
    # R_otimista = max(D_total, n*D_max - Z)
    R_lim_otimista = [max(D_total, n * D_max - Z) for n in n_clientes]
    # R_pessimista = n * D_total
    R_lim_pessimista = [n * D_total for n in n_clientes]

    df_sistema = pd.DataFrame({
        'n (Clientes)': n_clientes,
        'X_0': mva_X,
        'R': mva_R,
        'X Otimista': X_lim_otimista,
        'X Pessimista': X_lim_pessimista,
        'R Otimista': R_lim_otimista,
        'R Pessimista': R_lim_pessimista
    })

    df_sistema.to_csv('resultado.csv', index=False)

    # --- Gráficos ---

    sns.set_style("whitegrid")

    throughput_data = pd.melt(
        df_sistema,
        id_vars='n (Clientes)',
        value_vars=['X_0', 'X Otimista', 'X Pessimista'],
        var_name='Cenário',
        value_name='Throughput'
    )

    response_data = pd.melt(
        df_sistema,
        id_vars='n (Clientes)',
        value_vars=['R', 'R Otimista', 'R Pessimista'],
        var_name='Cenário',
        value_name='Tempo de Resposta'
    )

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

    # Gráfico do Throughput
    sns.lineplot(
        data=throughput_data,
        x='n (Clientes)',
        y='Throughput',
        hue='Cenário',
        style='Cenário',
        markers=True,
        dashes={'X_0': '', 'X Otimista': (3, 2), 'X Pessimista': (3, 2)},
        ax=axs[0]
    )
    axs[0].set(
        title='Throughput vs Número de Clientes (Limites)',
        xlabel='Número de Clientes (n)',
        ylabel='Throughput (transações/s)'
    )

    # Gráfico do Tempo de Resposta
    sns.lineplot(
        data=response_data,
        x='n (Clientes)',
        y='Tempo de Resposta',
        hue='Cenário',
        style='Cenário',
        markers=True,
        dashes={'R': '', 'R Otimista': (3, 2), 'R Pessimista': (3, 2)},
        ax=axs[1]
    )
    axs[1].set(
        title='Tempo de Resposta vs Número de Clientes (Limites)',
        xlabel='Número de Clientes (n)',
        ylabel='Tempo de Resposta (s)'
    )

    # Marca o ponto de saturação
    for ax in axs:
        ax.axvline(N_star, color='red', linestyle='--', linewidth=2, label=f'N* = {N_star:.2f}')
        ax.legend(frameon=True, title='Cenário')

    sns.despine()
    plt.tight_layout()
    plt.show()