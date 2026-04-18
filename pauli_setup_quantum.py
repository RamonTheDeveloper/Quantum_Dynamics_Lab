import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP DO ESTADO E MATRIZES DE PAULI
# ==========================================

# Definindo um estado inicial puro. 
# Vamos usar theta = pi/3 e phi = 0 (um estado no plano X-Z)
theta = np.pi / 3
phi = 0.0
# Criação do vetor de estado (Ket)
psi = np.cos(theta/2) * qt.basis(2, 0) + np.sin(theta/2) * np.exp(1j * phi) * qt.basis(2, 1)

# Matriz de densidade verdadeira (rho = |psi><psi|)
rho_true = psi * psi.dag()

# Matrizes de Pauli (nossas bases de medição X, Y, Z)
sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
I = qt.qeye(2) # Matriz Identidade

# ==========================================
# 2. SIMULAÇÃO DAS MEDIÇÕES
# ==========================================

def simulate_measurements(rho, operator, num_shots):
    """
    Simula N medições projetivas em um operador de Pauli.
    Retorna o valor esperado estimado e as contagens.
    """
    # Operadores de projeção para autovalores +1 e -1
    P_plus = (I + operator) / 2
    
    # A probabilidade de medir +1 é Tr(rho * P_plus)
    prob_plus = qt.expect(P_plus, rho)
    
    # Simula 'num_shots' usando uma distribuição binomial (como jogar moedas viciadas)
    counts_plus = np.random.binomial(num_shots, prob_plus)
    counts_minus = num_shots - counts_plus
    
    # Valor esperado estimado = (N(+) - N(-)) / N_total
    exp_val = (counts_plus - counts_minus) / num_shots
    
    return exp_val

# Número de medições experimentais por base
# Experimente mudar para 10, 100, 10000 para ver a convergência!
N_shots = 500

print(f"--- Iniciando Tomografia com {N_shots} medições por base ---")

# Simulando medições nas três bases
exp_x = simulate_measurements(rho_true, sx, N_shots)
exp_y = simulate_measurements(rho_true, sy, N_shots)
exp_z = simulate_measurements(rho_true, sz, N_shots)

print(f"<X> Estimado: {exp_x:+.4f} | Teórico: {qt.expect(sx, rho_true):+.4f}")
print(f"<Y> Estimado: {exp_y:+.4f} | Teórico: {qt.expect(sy, rho_true):+.4f}")
print(f"<Z> Estimado: {exp_z:+.4f} | Teórico: {qt.expect(sz, rho_true):+.4f}")

# ==========================================
# 3. RECONSTRUÇÃO DO ESTADO
# ==========================================

# A fórmula mágica da tomografia de 1 qubit:
# rho_reconstruido = 1/2 * (I + <X>X + <Y>Y + <Z>Z)
rho_reconstructed = 0.5 * (I + exp_x * sx + exp_y * sy + exp_z * sz)

# ==========================================
# 4. VISUALIZAÇÃO NA ESFERA DE BLOCH
# ==========================================

b = qt.Bloch()

# Adiciona o estado verdadeiro (linha sólida azul por padrão)
b.add_states(rho_true, 'vector')

# Adiciona o estado reconstruído (linha sólida vermelha)
b.add_states(rho_reconstructed, 'vector')
b.vector_color = ['#0055ff', '#ff0000'] 

print("\nGerando Esfera de Bloch...")
print("Vetor Azul: Estado Verdadeiro")
print("Vetor Vermelho: Estado Reconstruído via Estatística")

# Renderiza e mantém a janela aberta até o usuário fechá-la
b.render()
plt.show(block=True)