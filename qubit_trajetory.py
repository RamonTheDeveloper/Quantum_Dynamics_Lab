import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Parâmetros
omega_x = 1.0
omega_z = 0.3
t_max   = 2 * np.pi
tlist   = np.linspace(0, t_max, 100)

# Hamiltoniano e estado inicial
H    = omega_x * qt.sigmax() / 2 + omega_z * qt.sigmaz() / 2
psi0 = qt.basis(2, 0)

# Evolução com e_ops para obter observáveis
result = qt.sesolve(H, psi0, tlist,
                    e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])

sx = np.array(result.expect[0])
sy = np.array(result.expect[1])
sz = np.array(result.expect[2])

# Estado final: calculado direto pelo operador de evolução
U_final  = (1j * H * t_max).expm()   # U = e^{-iHt} ... sinal corrigido abaixo
U_final  = (-1j * H * t_max).expm()  # U(t) = exp(-iHt)
psi_final = U_final * psi0

# Esfera de Bloch
b = qt.Bloch()
b.figsize = [4, 4]        # ← tamanho menor
b.add_points(np.array([sx, sy, sz]), meth='l')
b.point_color = ['#534AB7']
b.point_size  = [4]

b.figsize = [4, 4]      # tamanho da figura em polegadas (padrão é [7, 7])
b.sphere_alpha = 0.1    # transparência da esfera (0 = invisível, 1 = sólida)

b.add_states(psi0)        # estado inicial
b.add_states(psi_final)   # estado final — sem depender de result.states

b.show()
b.axes.set_box_aspect([0.6, 0.6, 0.6])   # encolhe o box 3D
plt.suptitle(f'Trajetória do qubit  (ωₓ={omega_x}, ωᶻ={omega_z})',
             y=1.00, fontsize=15)
plt.savefig('bloch_trajectory.png', dpi=80, bbox_inches='tight')
plt.show()