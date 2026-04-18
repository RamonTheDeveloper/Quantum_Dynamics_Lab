import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Parâmetros
omega   = 1.0
Omega_R = 1.0
t_max   = 6 * np.pi / Omega_R
tlist   = np.linspace(0, t_max, 500)
psi0    = qt.basis(2, 1)
P_exc   = qt.basis(2, 0) * qt.basis(2, 0).dag()
detunings = [0.0, 0.5, 1.0, 2.0]
colors    = ['#534AB7', '#1D9E75', '#D85A30', '#888780']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Painel 1: Oscilações de Rabi
for d, c in zip(detunings, colors):
    H_d = d * qt.sigmaz() / 2 + Omega_R * qt.sigmax() / 2
    res = qt.sesolve(H_d, psi0, tlist, e_ops=[P_exc])
    ax1.plot(tlist * Omega_R / np.pi, res.expect[0],
             color=c, label=f'δ={d}')

ax1.set_xlabel('Tempo (π/Ω_R)')
ax1.set_ylabel('P(excitado)')
ax1.set_title('Oscilações de Rabi')
ax1.legend()

# Painel 2: Espectro de Rabi
# Tempo adaptativo — cobre pelo menos 2 ciclos de Rabi generalizado
delta_sweep = np.linspace(-5, 5, 200)
P_max = []

for d in delta_sweep:
    Omega_gen = np.sqrt(Omega_R**2 + d**2)   # frequência de Rabi generalizada
    t_end     = 4 * np.pi / Omega_gen           # 2 ciclos completos
    tl        = np.linspace(0, t_end, 100)     # sempre 100 pontos

    H_d = d * qt.sigmaz() / 2 + Omega_R * qt.sigmax() / 2
    res = qt.sesolve(H_d, psi0, tl, e_ops=[P_exc])
    P_max.append(max(res.expect[0]))

ax2.plot(delta_sweep / Omega_R, P_max, color='#534AB7')
ax2.fill_between(delta_sweep / Omega_R, P_max, alpha=0.15, color='#534AB7')
ax2.set_xlabel('Detuning δ/Ω_R')
ax2.set_ylabel('P_max (excitado)')
ax2.set_title('Espectro de Rabi')

plt.tight_layout()
plt.savefig('rabi.png', dpi=150, bbox_inches='tight')
plt.show()