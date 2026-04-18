import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# 1. Parâmetros do Sistema
# Frequência de Rabi (representa a força do pulso eletromagnético)
omega = 1.0 * 2 * np.pi 
# Lista de tempo: de 0 a 3 segundos, com 100 pontos
tlist = np.linspace(0, 3, 100) 

# 2. Construindo a Física
# Operador de Pauli X (representa a interação com o campo)
sx = qt.sigmax()
# Operador de Pauli Z (representa a medição da energia)
sz = qt.sigmaz()

# O Hamiltoniano do sistema (A energia total)
H = 0.5 * omega * sx

# 3. Estado Inicial
# Definimos que o sistema começa 100% no estado fundamental (baixo)
estado_inicial = qt.basis(2, 1) # No QuTiP, basis(2,1) é o estado |0> e basis(2,0) é o |1>

# 4. Resolvendo a Equação de Schrödinger
# A função 'mesolve' do QuTiP é a estrela aqui. Ela calcula a evolução do tempo.
# Pedimos para ela calcular o valor esperado de Z (para saber a probabilidade)
resultado = qt.mesolve(H, estado_inicial, tlist, [], [sz])

# 5. Organizando os dados para o gráfico
# Convertendo o valor esperado de Z em Probabilidade (de 0 a 1)
prob_excitado = (resultado.expect[0] + 1) / 2
prob_fundamental = 1 - prob_excitado

# 6. Plotando com Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(tlist, prob_excitado, label='Estado Excitado $|1\\rangle$', color='red', lw=2)
plt.plot(tlist, prob_fundamental, label='Estado Fundamental $|0\\rangle$', color='blue', linestyle='--', lw=2)

plt.title('Oscilações de Rabi: Interação Qubit-Campo Eletromagnético', fontsize=14)
plt.xlabel('Tempo (s)', fontsize=12)
plt.ylabel('Probabilidade', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Mostra o gráfico na tela
plt.show()
