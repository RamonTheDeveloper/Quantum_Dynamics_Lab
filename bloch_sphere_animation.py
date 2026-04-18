import sys
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

sys.stdout.reconfigure(encoding='utf-8')

# Parâmetros
omega_x = 1.0       # frequencia de Rabi (eixo x)
omega_z = 0.4       # detuning (eixo z)
t_max   = 4 * np.pi
n_frames = 120      # quantidade de frames da animacao
trail_len = 30      # comprimento da trilha (frames anteriores visiveis)

# Hamiltoniano e Evolução
H     = omega_x * qt.sigmax() / 2 + omega_z * qt.sigmaz() / 2
psi0  = qt.basis(2, 0)
tlist = np.linspace(0, t_max, n_frames)

result = qt.sesolve(H, psi0, tlist,
                    e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])

sx = np.array(result.expect[0])
sy = np.array(result.expect[1])
sz = np.array(result.expect[2])

# Figura e esfera base
fig = plt.figure(figsize=(6, 6))
fig.patch.set_facecolor('#0d0d0d')

b = qt.Bloch(fig=fig)
b.figsize      = [6, 6]
b.sphere_color = '#1a1a2e'
b.sphere_alpha = 0.15
b.frame_color  = '#444466'
b.frame_alpha  = 0.3
b.axis_label   = ['x', 'y', 'z']
b.font_color   = 'white'
b.font_size    = 14

b.show()
ax = b.axes
ax.set_facecolor('#0d0d0d')

# Elementos animados
trail_line, = ax.plot([], [], [], color='#7F77DD',
                      linewidth=1.2, alpha=0.6)
head_dot,   = ax.plot([], [], [], 'o',
                      color='#EF9F27', markersize=8, zorder=10)
vector_line, = ax.plot([], [], [],
                       color='#EF9F27', linewidth=2.5, zorder=9)

# Label de tempo
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                      color='white', fontsize=10)


def init():
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    head_dot.set_data([], [])
    head_dot.set_3d_properties([])
    vector_line.set_data([], [])
    vector_line.set_3d_properties([])
    time_text.set_text('')
    return trail_line, head_dot, vector_line, time_text


def update(frame):
    # trilha: ultimos trail_len frames
    start = max(0, frame - trail_len)
    xs = sx[start:frame+1]
    ys = sy[start:frame+1]
    zs = sz[start:frame+1]

    trail_line.set_data(xs, ys)
    trail_line.set_3d_properties(zs)

    # ponto atual
    head_dot.set_data([sx[frame]], [sy[frame]])
    head_dot.set_3d_properties([sz[frame]])

    # vetor da origem ate o ponto atual
    vector_line.set_data([0, sx[frame]], [0, sy[frame]])
    vector_line.set_3d_properties([0, sz[frame]])

    # rotacao lenta da camera
    ax.view_init(elev=20, azim=frame * 1.5)

    t = tlist[frame]
    time_text.set_text(f't = {t:.2f}')

    return trail_line, head_dot, vector_line, time_text


ani = FuncAnimation(fig, update, frames=n_frames,
                    init_func=init, interval=40, blit=False)

# Salvar
print("Salvando animacao... pode demorar alguns segundos.")

# salva como GIF
ani.save('bloch_animation.gif',
         writer=PillowWriter(fps=25),
         dpi=120)

print("Salvo como bloch_animation.gif")
print("Para salvar como MP4 instale ffmpeg e use:")
print("  ani.save('bloch_animation.mp4', writer='ffmpeg', fps=25, dpi=150)")

plt.show()