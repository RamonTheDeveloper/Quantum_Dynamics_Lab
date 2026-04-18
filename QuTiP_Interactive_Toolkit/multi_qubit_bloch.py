import sys
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

sys.stdout.reconfigure(encoding='utf-8')

# ── parametros ───────────────────────────────────────────
t_max    = 4 * np.pi
n_frames = 150
trail_len = 40

# ── 3 qubits com Hamiltonianos diferentes ────────────────
#    cada um tem frequencia e eixo de rotacao distintos
QUBITS = [
    {
        'label': 'Qubit 1',
        'H':     1.0 * qt.sigmax() / 2,               # rotacao pura em X
        'psi0':  qt.basis(2, 0),                       # inicia em |0>
        'color': '#EF9F27',                            # laranja
    },
    {
        'label': 'Qubit 2',
        'H':     0.6 * qt.sigmax() / 2
               + 0.8 * qt.sigmaz() / 2,               # rotacao inclinada
        'psi0':  (qt.basis(2, 0) + qt.basis(2, 1)).unit(),  # inicia em |+>
        'color': '#5DCAA5',                            # verde-azulado
    },
    {
        'label': 'Qubit 3',
        'H':     0.4 * qt.sigmay() / 2
               + 0.9 * qt.sigmaz() / 2,               # rotacao em Y+Z
        'psi0':  qt.basis(2, 1),                       # inicia em |1>
        'color': '#ED93B1',                            # rosa
    },
]

# ── evolucao de cada qubit ────────────────────────────────
tlist = np.linspace(0, t_max, n_frames)

for q in QUBITS:
    res = qt.sesolve(q['H'], q['psi0'], tlist,
                     e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])
    q['sx'] = np.array(res.expect[0])
    q['sy'] = np.array(res.expect[1])
    q['sz'] = np.array(res.expect[2])

    # estado final via operador de evolucao (para ponto final)
    U_final   = (-1j * q['H'] * t_max).expm()
    q['psi_f'] = U_final * q['psi0']

# ── figura ────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 7))
fig.patch.set_facecolor('#0d0d0d')

b = qt.Bloch(fig=fig)
b.sphere_color = '#111133'
b.sphere_alpha = 0.12
b.frame_color  = '#333355'
b.frame_alpha  = 0.25
b.axis_label   = ['x', 'y', 'z']
b.font_color   = '#aaaacc'
b.font_size    = 13
b.show()

ax = b.axes
ax.set_facecolor('#0d0d0d')

# ── elementos animados por qubit ──────────────────────────
artists = []
for q in QUBITS:
    trail,  = ax.plot([], [], [], color=q['color'],
                      linewidth=1.2, alpha=0.5)
    vector, = ax.plot([], [], [], color=q['color'],
                      linewidth=2.8, zorder=9)
    dot,    = ax.plot([], [], [], 'o', color=q['color'],
                      markersize=9, zorder=10,
                      markeredgecolor='white', markeredgewidth=0.5)
    artists.append((trail, vector, dot))

# label de tempo
time_text = ax.text2D(0.04, 0.95, '', transform=ax.transAxes,
                      color='white', fontsize=10)

# legenda manual
for i, q in enumerate(QUBITS):
    ax.text2D(0.04, 0.88 - i * 0.07, q['label'],
              transform=ax.transAxes,
              color=q['color'], fontsize=10, fontweight='bold')


def init():
    for trail, vector, dot in artists:
        for obj in (trail, vector, dot):
            obj.set_data([], [])
            obj.set_3d_properties([])
    time_text.set_text('')
    return [a for group in artists for a in group] + [time_text]


def update(frame):
    for i, (q, (trail, vector, dot)) in enumerate(zip(QUBITS, artists)):
        # trilha
        start = max(0, frame - trail_len)
        xs = q['sx'][start:frame+1]
        ys = q['sy'][start:frame+1]
        zs = q['sz'][start:frame+1]
        trail.set_data(xs, ys)
        trail.set_3d_properties(zs)

        # vetor da origem ao ponto atual
        cx, cy, cz = q['sx'][frame], q['sy'][frame], q['sz'][frame]
        vector.set_data([0, cx], [0, cy])
        vector.set_3d_properties([0, cz])

        # ponto na ponta
        dot.set_data([cx], [cy])
        dot.set_3d_properties([cz])

    # rotacao lenta da camera
    ax.view_init(elev=22, azim=frame * 1.2)
    time_text.set_text(f't = {tlist[frame]:.2f}')

    return [a for group in artists for a in group] + [time_text]


ani = FuncAnimation(fig, update, frames=n_frames,
                    init_func=init, interval=40, blit=False)

print("Salvando GIF... aguarde.")
ani.save('multi_qubit_bloch.gif',
         writer=PillowWriter(fps=25),
         dpi=120)
print("Salvo: multi_qubit_bloch.gif")

plt.show()