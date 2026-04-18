import sys
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.stdout.reconfigure(encoding='utf-8')

# ════════════════════════════════════════════════════════
#  CONFIGURE AQUI MANUALMENTE
# ════════════════════════════════════════════════════════
SEQUENCE   = ['H', 'T', 'T', 'S', 'X', 'H']   # sequencia de portas
INIT_STATE = '0'                                # '0', '1', '+', '-'
FRAMES_PER_GATE  = 40    # frames de rotacao por porta
FRAMES_PAUSE     = 18    # frames de pausa entre portas
CAMERA_SPIN      = True  # camera gira lentamente
SAVE_GIF         = True  # salva como GIF
GIF_FPS          = 60
# ════════════════════════════════════════════════════════

# Estados iniciais
INIT_STATES = {
    '0': qt.basis(2, 0),
    '1': qt.basis(2, 1),
    '+': (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
    '-': (qt.basis(2, 0) - qt.basis(2, 1)).unit(),
}

# Definicao das portas
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

GATE_COLORS = {
    'X': '#E24B4A', 'Y': '#5DCAA5', 'Z': '#378ADD',
    'H': '#EF9F27', 'S': '#ED93B1', 'T': '#7F77DD',
    'RX90': '#D85A30', 'RY90': '#1D9E75',
}

GATE_DESC = {
    'X': 'Porta X  — rotacao 180 em torno de x',
    'Y': 'Porta Y  — rotacao 180 em torno de y',
    'Z': 'Porta Z  — rotacao 180 em torno de z',
    'H': 'Hadamard — rotacao 180 em torno de (x+z)',
    'S': 'Porta S  — rotacao 90 em torno de z',
    'T': 'Porta T  — rotacao 45 em torno de z',
    'RX90': 'Rx(90)   — rotacao 90 em torno de x',
    'RY90': 'Ry(90)   — rotacao 90 em torno de y',
}

# Pre-calcula trajetorias de cada porta
def gate_trajectory(G, psi, n=40):
    """Trajetoria suave de aplicar porta G partindo de psi."""
    H_eff = 1j * G.logm()
    pts   = []
    for t in np.linspace(0, 1, n):
        U     = (-1j * H_eff * t).expm()
        psi_t = U * psi
        pts.append([
            float(qt.expect(qt.sigmax(), psi_t).real),
            float(qt.expect(qt.sigmay(), psi_t).real),
            float(qt.expect(qt.sigmaz(), psi_t).real),
        ])
    return np.array(pts), G * psi

def get_xyz(psi):
    return (
        float(qt.expect(qt.sigmax(), psi).real),
        float(qt.expect(qt.sigmay(), psi).real),
        float(qt.expect(qt.sigmaz(), psi).real),
    )

print("Pre-calculando trajetorias...")
psi = INIT_STATES[INIT_STATE]
segments = []   # cada item: (gate_name, traj_array, psi_inicial)
for name in SEQUENCE:
    G    = GATES[name]
    traj, psi_next = gate_trajectory(G, psi, n=FRAMES_PER_GATE)
    segments.append({
        'name':   name,
        'traj':   traj,
        'psi_in': psi,
        'color':  GATE_COLORS.get(name, '#ffffff'),
    })
    psi = psi_next
print(f"  {len(segments)} portas calculadas.")

# Monta a lista global de frames
# Cada frame e uma tupla (tipo, seg_idx, frame_local)
# tipo: 'rotate' ou 'pause'
frame_list = []
for i, seg in enumerate(segments):
    for f in range(FRAMES_PER_GATE):
        frame_list.append(('rotate', i, f))
    for f in range(FRAMES_PAUSE):
        frame_list.append(('pause', i, f))

N_FRAMES = len(frame_list)

# Historico de pontos ja percorridos
# pre-calcula posicao (x,y,z) em cada frame global
all_positions = []
for ftype, seg_idx, flocal in frame_list:
    seg = segments[seg_idx]
    if ftype == 'rotate':
        pt = seg['traj'][flocal]
    else:
        pt = seg['traj'][-1]
    all_positions.append(pt)
all_positions = np.array(all_positions)

# Figura
fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor('#0d0d0d')

# painel esfera (esquerda)
b = qt.Bloch(fig=fig)
b.sphere_color = '#111133'
b.sphere_alpha = 0.12
b.frame_color  = '#333355'
b.frame_alpha  = 0.25
b.font_color   = '#aaaacc'
b.font_size    = 12
b.show()
ax = b.axes
ax.set_facecolor('#0d0d0d')
ax.set_position([0.0, 0.1, 0.75, 0.85])

# painel lateral: circuito
ax2 = fig.add_axes([0.76, 0.1, 0.22, 0.85])
ax2.set_facecolor('#0d0d0d')
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, len(SEQUENCE) - 0.5)
ax2.axis('off')
ax2.set_title('Circuito', color='#aaaacc', fontsize=11, pad=8)

# desenha boxes do circuito (estaticos)
gate_boxes = []
gate_texts = []
for i, name in enumerate(SEQUENCE):
    y   = len(SEQUENCE) - 1 - i
    col = GATE_COLORS.get(name, '#ffffff')
    box = mpatches.FancyBboxPatch(
        (0.15, y - 0.3), 0.7, 0.6,
        boxstyle='round,pad=0.05',
        facecolor='#1a1a2e',
        edgecolor=col, linewidth=1.5,
        transform=ax2.transData, zorder=2
    )
    ax2.add_patch(box)
    txt = ax2.text(0.5, y, name, ha='center', va='center',
                   color=col, fontsize=13, fontweight='bold', zorder=3)
    gate_boxes.append(box)
    gate_texts.append(txt)

# seta indicadora no circuito
indicator = ax2.annotate(
    '', xy=(0.15, len(SEQUENCE) - 0.5),
    xytext=(-0.05, len(SEQUENCE) - 0.5),
    arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
)

# Elementos animados na esfera
# trilha historica (todos os segmentos ja completos)
history_lines = []
for seg in segments:
    ln, = ax.plot([], [], [], linewidth=1.0, alpha=0.35,
                  color=seg['color'])
    history_lines.append(ln)

# trilha do segmento atual
current_trail, = ax.plot([], [], [], linewidth=1.8, alpha=0.8,
                         color='white')

# vetor atual
vector_line, = ax.plot([], [], [], linewidth=3.0, zorder=9,
                       color='white')
head_dot,    = ax.plot([], [], [], 'o', markersize=10, zorder=10,
                       color='white', markeredgecolor='#cccccc',
                       markeredgewidth=0.5)

# label de porta ativa
gate_label = ax.text2D(0.03, 0.06, '', transform=ax.transAxes,
                       color='white', fontsize=10,
                       bbox=dict(facecolor='#1a1a2e', alpha=0.7,
                                 edgecolor='#444466', pad=4))

# label de estado
state_label = ax.text2D(0.03, 0.95, '', transform=ax.transAxes,
                        color='#aaaacc', fontsize=9)

# Animacao
completed_segs = set()

def init():
    for ln in history_lines:
        ln.set_data([], [])
        ln.set_3d_properties([])
    for obj in (current_trail, vector_line, head_dot):
        obj.set_data([], [])
        obj.set_3d_properties([])
    gate_label.set_text('')
    state_label.set_text('')
    return (history_lines + [current_trail, vector_line,
                             head_dot, gate_label, state_label])

def update(frame):
    ftype, seg_idx, flocal = frame_list[frame]
    seg   = segments[seg_idx]
    color = seg['color']
    name  = seg['name']

    # trilhas de segmentos ja concluidos
    for i in range(seg_idx):
        t = segments[i]['traj']
        history_lines[i].set_data(t[:, 0], t[:, 1])
        history_lines[i].set_3d_properties(t[:, 2])

    # trilha do segmento atual
    if ftype == 'rotate':
        t = seg['traj'][:flocal+1]
    else:
        t = seg['traj']
    current_trail.set_data(t[:, 0], t[:, 1])
    current_trail.set_3d_properties(t[:, 2])
    current_trail.set_color(color)

    # posicao atual
    pt = all_positions[frame]
    vector_line.set_data([0, pt[0]], [0, pt[1]])
    vector_line.set_3d_properties([0, pt[2]])
    vector_line.set_color(color)
    head_dot.set_data([pt[0]], [pt[1]])
    head_dot.set_3d_properties([pt[2]])
    head_dot.set_color(color)

    # camera
    if CAMERA_SPIN:
        ax.view_init(elev=22, azim=frame * 0.8)

    # label da porta
    if ftype == 'rotate':
        gate_label.set_text(GATE_DESC.get(name, name))
        gate_label.set_color(color)
    else:
        gate_label.set_text('...')
        gate_label.set_color('#888888')

    # estado atual (coordenadas)
    state_label.set_text(
        f'<X>={pt[0]:.2f}  <Y>={pt[1]:.2f}  <Z>={pt[2]:.2f}'
    )

    # indicador no circuito
    y_arrow = len(SEQUENCE) - 1 - seg_idx
    indicator.xy     = (0.15, y_arrow)
    indicator.xyann  = (-0.05, y_arrow)

    # destaque no circuito
    for i, (box, txt) in enumerate(zip(gate_boxes, gate_texts)):
        if i == seg_idx and ftype == 'rotate':
            box.set_edgecolor(GATE_COLORS.get(SEQUENCE[i], '#fff'))
            box.set_linewidth(2.5)
            box.set_facecolor('#2a2a4e')
        elif i < seg_idx:
            box.set_facecolor('#0d1a0d')
            box.set_linewidth(1.0)
        else:
            box.set_facecolor('#1a1a2e')
            box.set_linewidth(1.0)

    return (history_lines + [current_trail, vector_line,
                             head_dot, gate_label, state_label])


ani = FuncAnimation(fig, update, frames=N_FRAMES,
                    init_func=init, interval=40, blit=False)

if SAVE_GIF:
    print(f"Salvando GIF ({N_FRAMES} frames)... aguarde.")
    ani.save('gates_animation.gif',
             writer=PillowWriter(fps=GIF_FPS),
             dpi=110)
    print("Salvo: gates_animation.gif")

plt.show()