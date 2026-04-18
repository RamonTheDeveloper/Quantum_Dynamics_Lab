import sys
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.stdout.reconfigure(encoding='utf-8')

# Portas quanticas
GATES = {
    'X':    qt.sigmax(),
    'Y':    qt.sigmay(),
    'Z':    qt.sigmaz(),
    'H':    (qt.sigmax() + qt.sigmaz()).unit(),
    'S':    (-1j * np.pi / 4 * qt.sigmaz()).expm(),
    'T':    (-1j * np.pi / 8 * qt.sigmaz()).expm(),
    'RX90': (-1j * np.pi / 4 * qt.sigmax()).expm(),
    'RY90': (-1j * np.pi / 4 * qt.sigmay()).expm(),
}

# Estados iniciais
INIT_STATES = {
    '0': qt.basis(2, 0),
    '1': qt.basis(2, 1),
    '+': (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
    '-': (qt.basis(2, 0) - qt.basis(2, 1)).unit(),
}


def get_xyz(psi):
    """Retorna coordenadas (x,y,z) de Bloch de um estado."""
    return (
        float(qt.expect(qt.sigmax(), psi).real),
        float(qt.expect(qt.sigmay(), psi).real),
        float(qt.expect(qt.sigmaz(), psi).real),
    )


def apply_gate(gate_name, psi):
    G     = GATES[gate_name]
    H_eff = 1j * G.logm()
    traj  = []
    for t in np.linspace(0, 1, 40):
        U     = (-1j * H_eff * t).expm()
        psi_t = U * psi
        traj.append(get_xyz(psi_t))
    return np.array(traj), G * psi


def show_state_info(psi):
    x, y, z = get_xyz(psi)
    print(f'  <X>={x:.3f}  <Y>={y:.3f}  <Z>={z:.3f}')


def plot_circuit(sequence, psi0):
    raw_colors = cm.plasma(np.linspace(0.1, 0.9, len(sequence)))
    hex_colors = [mcolors.to_hex(c) for c in raw_colors]

    psi       = psi0
    all_trajs = []
    for i, name in enumerate(sequence):
        traj, psi = apply_gate(name, psi)
        all_trajs.append((traj, hex_colors[i]))

    x0, y0, z0 = get_xyz(psi0)
    xf, yf, zf = get_xyz(psi)

    # Esfera vazia
    b = qt.Bloch()
    b.figsize      = [5, 5]
    b.sphere_alpha = 0.08
    b.frame_alpha  = 0.15
    b.show()

    ax = b.axes

    # Trajetorias coloridas
    for traj, color in all_trajs:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, linewidth=2)

    # Ponto inicial (verde) e final (vermelho) manualmente
    ax.scatter([x0], [y0], [z0], color='#1D9E75', s=60, zorder=5)
    ax.scatter([xf], [yf], [zf], color='#D85A30', s=60, zorder=5)

    handles = [
        plt.Line2D([0], [0], color=hex_colors[i], lw=2,
                   label=f'Passo {i+1}: {sequence[i]}')
        for i in range(len(sequence))
    ]
    handles += [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#1D9E75', markersize=8, label='Inicial'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#D85A30', markersize=8, label='Final'),
    ]
    plt.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)
    plt.suptitle(' -> '.join(sequence), fontsize=11, y=0.98)
    plt.savefig('circuit_bloch.png', dpi=150, bbox_inches='tight')
    plt.show()


# loop principal
print("=== Simulador de portas - Esfera de Bloch ===")

while True:
    print("\nEstados iniciais: 0  1  +  -")
    init = input("Estado inicial [0]: ").strip() or '0'
    psi0 = INIT_STATES.get(init, INIT_STATES['0'])

    print("\nPortas: X, Y, Z, H, S, T, RX90 e RY90")
    print("Separe por espaco. Ex: H X H")
    raw      = input("Sequencia de portas: ").strip().upper()
    sequence = [g for g in raw.split() if g in GATES]

    if not sequence:
        print("Nenhuma porta válida. Tente novamente.")
        continue

    print(f"\nAplicando: {' -> '.join(sequence)}")
    print("  Estado inicial:", end='')
    show_state_info(psi0)

    psi = psi0
    for g in sequence:
        _, psi = apply_gate(g, psi)
        print(f"  apos {g}:", end='')
        show_state_info(psi)

    plot_circuit(sequence, psi0)

    if input("\nNova sequência? [s/n]: ").strip().lower() != 's':
        break

print("Até mais!")