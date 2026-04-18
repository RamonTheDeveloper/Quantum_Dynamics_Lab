import sys
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

sys.stdout.reconfigure(encoding='utf-8')

# ════════════════════════════════════════════════════════
#  CONFIGURE AQUI
# ════════════════════════════════════════════════════════
# Estado a ser teleportado (angulos na esfera de Bloch)
THETA = np.pi / 3      # polar    (0=norte, pi=sul)
PHI   = np.pi / 4      # azimutal

FRAMES_PER_PHASE = 35  # frames por fase do protocolo
FRAMES_PAUSE     = 18  # pausa entre fases
SAVE_GIF         = True
GIF_FPS          = 60
# ════════════════════════════════════════════════════════

# Estado a teleportar
# |psi> = cos(theta/2)|0> + e^{i*phi} sin(theta/2)|1>
psi_in = (np.cos(THETA / 2) * qt.basis(2, 0)
          + np.exp(1j * PHI) * np.sin(THETA / 2) * qt.basis(2, 1))
psi_in = psi_in.unit()

def bloch_vec(psi):
    return np.array([
        float(qt.expect(qt.sigmax(), psi).real),
        float(qt.expect(qt.sigmay(), psi).real),
        float(qt.expect(qt.sigmaz(), psi).real),
    ])

# Operadores uteis
I2   = qt.identity(2)
H_op = (qt.Qobj(np.array([[1, 1], [1, -1]], dtype=complex)) / np.sqrt(2))
X_op = qt.sigmax()
Z_op = qt.sigmaz()

def tensor2(a, b): return qt.tensor(a, b)
def tensor3(a, b, c): return qt.tensor(a, b, c)

# Protocolo de teleportacao
# qubits: [0]=Alice_msg, [1]=Alice_epr, [2]=Bob_epr
# estado inicial: psi_in x |00>
psi_00 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
state0 = qt.tensor(psi_in, qt.basis(2, 0), qt.basis(2, 0))

# Passo 1: cria Estado de Bell entre qubits 1 e 2
# H no qubit 1, CNOT(1,2)
H1  = tensor3(I2, H_op, I2)

# CNOT entre qubits 1 e 2 no espaco de 3 qubits
def cnot_ij(i, j, N=3):
    """CNOT com controle i e alvo j em cadeia de N qubits."""
    ops_list = []
    # proj |0><0| no controle -> I no alvo
    P0 = qt.basis(2,0)*qt.basis(2,0).dag()
    P1 = qt.basis(2,1)*qt.basis(2,1).dag()
    ids0 = [I2]*N; ids0[i] = P0
    ids1 = [I2]*N; ids1[i] = P1; ids1[j] = X_op
    return qt.tensor(ids0) + qt.tensor(ids1)

H1_3q  = tensor3(I2,   H_op, I2)
C12_3q = cnot_ij(1, 2, 3)
C01_3q = cnot_ij(0, 1, 3)
H0_3q  = tensor3(H_op, I2,   I2)

# estado apos Bell state
state1 = C12_3q * H1_3q * state0

# Passo 2: Bell measurement de Alice (qubits 0 e 1)
state2 = H0_3q * C01_3q * state1

# Projecoes de medicao (resultados classicos 00, 01, 10, 11)
def proj_meas(result_bits):
    """Projeta qubits 0,1 no resultado dado e retorna estado normalizado de Bob."""
    b0, b1 = result_bits
    P = tensor3(
        qt.basis(2,b0)*qt.basis(2,b0).dag(),
        qt.basis(2,b1)*qt.basis(2,b1).dag(),
        I2
    )
    projected = P * state2
    norm = projected.norm()
    if norm < 1e-10:
        return None, 0.0
    return projected.unit(), float(norm**2)

# Passo 3: correcoes de Bob
def correction(b0, b1, psi3):
    """Aplica X^b1 Z^b0 no qubit de Bob."""
    ops = [I2, I2, I2]
    U = qt.tensor([I2, I2, I2])
    if b1 == 1:
        ops2 = [I2, I2, X_op]
        U = qt.tensor(ops2) * U
    if b0 == 1:
        ops3 = [I2, I2, Z_op]
        U = qt.tensor(ops3) * U
    return U * psi3

# Calcula os 4 resultados possiveis
results = {}
for b0 in range(2):
    for b1 in range(2):
        psi3, prob = proj_meas((b0, b1))
        if psi3 is not None:
            psi4 = correction(b0, b1, psi3)
            # traca qubit de Bob (qubit 2)
            rho4 = psi4 * psi4.dag()
            rho_bob = rho4.ptrace([2])
            # estado de Bob como vetor puro (approx)
            evals, evecs = rho_bob.eigenstates()
            psi_bob = evecs[-1]  # autovetor de maior autovalor
            results[(b0,b1)] = {
                'prob':    prob,
                'psi_bob': psi_bob,
                'bloch':   bloch_vec(psi_bob),
            }

# Resultado mais provavel
best = max(results, key=lambda k: results[k]['prob'])
psi_bob_final = results[best]['psi_bob']

print("Resultados da medicao:")
for k, v in results.items():
    print(f"  {k}: prob={v['prob']:.3f}  bloch={v['bloch']}")
print(f"  Estado entrada: {bloch_vec(psi_in)}")
print(f"  Melhor resultado: {best} -> fidelidade aprox")

# Trajetorias suaves para animacao
def smooth_traj(psi_start, psi_end, n=35):
    """Interpola suavemente entre dois estados na esfera de Bloch."""
    v0 = bloch_vec(psi_start)
    v1 = bloch_vec(psi_end)
    pts = []
    for t in np.linspace(0, 1, n):
        # interpolacao esferica (slerp simplificado)
        alpha = psi_start * np.cos(t * np.pi / 2) + psi_end * np.sin(t * np.pi / 2)
        if alpha.norm() > 1e-10:
            alpha = alpha.unit()
        pts.append(bloch_vec(alpha))
    return np.array(pts)

def rotate_traj(psi, H_rot, t_total, n=35):
    """Trajetoria de rotacao sob Hamiltoniano."""
    pts = []
    for t in np.linspace(0, t_total, n):
        U = (-1j * H_rot * t).expm()
        pts.append(bloch_vec(U * psi))
    return np.array(pts)

# trajetorias de cada qubit por fase
# fase 0: estado inicial (psi_in visivel, outros em |0>)
v_in  = bloch_vec(psi_in)
v_0   = bloch_vec(qt.basis(2, 0))

# fase 1: criacao do Bell state (qubits 1 e 2 se entrelaçam)
# mostra rotacao H no qubit 1
traj_bell_1 = rotate_traj(qt.basis(2,0),
                           qt.sigmax()/2, np.pi/2, FRAMES_PER_PHASE)
traj_bell_2 = rotate_traj(qt.basis(2,0),
                           qt.sigmay()/2, np.pi/2, FRAMES_PER_PHASE)

# fase 2: medicao de Bell (Alice aplica CNOT e H)
# qubit 0 "desce" para estado misto (representado indo ao centro)
traj_meas_0 = np.array([v_in * (1 - t/FRAMES_PER_PHASE)
                         for t in range(FRAMES_PER_PHASE)])
traj_meas_1 = np.array([traj_bell_1[-1] * (1 - t/FRAMES_PER_PHASE)
                         for t in range(FRAMES_PER_PHASE)])

# fase 3: correcao e verificacao (Bob recebe estado)
traj_bob = smooth_traj(qt.basis(2,0), psi_bob_final, FRAMES_PER_PHASE)

# ── lista de fases ────────────────────────────────────────
PHASES = [
    ('init',     'Estado inicial',          '#aaaacc'),
    ('bell',     'Criando Bell state',      '#5DCAA5'),
    ('pause1',   '...',                     '#444466'),
    ('meas',     'Medicao de Bell',         '#EF9F27'),
    ('pause2',   '...',                     '#444466'),
    ('classical','Canal classico',          '#ED93B1'),
    ('pause3',   '...',                     '#444466'),
    ('correct',  'Correcao de Bob',         '#7F77DD'),
    ('verify',   'Verificacao',             '#E24B4A'),
]

frame_list = []
for ph, label, color in PHASES:
    n = FRAMES_PAUSE if 'pause' in ph else FRAMES_PER_PHASE
    for f in range(n):
        frame_list.append((ph, f, label, color))

N_FRAMES = len(frame_list)

# posicao de cada qubit em cada frame
def get_positions(frame):
    ph, idx, _, _ = frame_list[frame]
    n = FRAMES_PER_PHASE

    if ph == 'init':
        t = idx / max(n-1, 1)
        # psi_in aparece suavemente
        scale = min(1.0, t * 3)
        return (v_in * scale, v_0 * 0.05, v_0 * 0.05)

    elif ph == 'bell':
        t = idx / max(n-1, 1)
        # qubits 1 e 2 se movem para o equador
        q1 = traj_bell_1[idx] if idx < len(traj_bell_1) else traj_bell_1[-1]
        q2 = traj_bell_2[idx] if idx < len(traj_bell_2) else traj_bell_2[-1]
        return (v_in, q1, q2)

    elif ph == 'pause1':
        return (v_in, traj_bell_1[-1], traj_bell_2[-1])

    elif ph == 'meas':
        t = idx / max(n-1, 1)
        q0 = v_in   * (1 - t * 0.8)
        q1 = traj_bell_1[-1] * (1 - t * 0.8)
        q2 = traj_bell_2[-1]
        return (q0, q1, q2)

    elif ph == 'pause2':
        return (v_in * 0.2, traj_bell_1[-1] * 0.2, traj_bell_2[-1])

    elif ph == 'classical':
        t  = idx / max(n-1, 1)
        q2 = traj_bob[min(int(t * len(traj_bob) * 0.3), len(traj_bob)-1)]
        return (v_in * 0.2, traj_bell_1[-1] * 0.2, q2)

    elif ph == 'pause3':
        return (v_in * 0.2, traj_bell_1[-1] * 0.2, traj_bob[len(traj_bob)//3])

    elif ph == 'correct':
        i = min(idx, len(traj_bob)-1)
        return (v_in * 0.2, traj_bell_1[-1] * 0.2, traj_bob[i])

    else:  # verify
        t  = idx / max(n-1, 1)
        scale = 1.0 + 0.08 * np.sin(t * np.pi * 4)  # pulsa levemente
        q2 = bloch_vec(psi_bob_final) * scale
        return (v_in * 0.2, traj_bell_1[-1] * 0.2, q2)

all_pos = [get_positions(f) for f in range(N_FRAMES)]

# Figura
fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor('#0d0d0d')

gs = gridspec.GridSpec(2, 4, figure=fig,
                        height_ratios=[1.6, 1],
                        width_ratios=[2.2, 1, 1, 1],
                        hspace=0.4, wspace=0.3)

ax_bloch  = fig.add_subplot(gs[:, 0])
ax_circ   = fig.add_subplot(gs[0, 1:3])
ax_fid    = fig.add_subplot(gs[0, 3])
ax_prob   = fig.add_subplot(gs[1, 1])
ax_bits   = fig.add_subplot(gs[1, 2])
ax_info   = fig.add_subplot(gs[1, 3])

for ax in [ax_circ, ax_fid, ax_prob, ax_bits, ax_info]:
    ax.set_facecolor('#0d0d0d')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_title(title,   color='#aaaacc', fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color='#888899', fontsize=8)
    ax.set_ylabel(ylabel, color='#888899', fontsize=8)
    ax.tick_params(colors='#888899', labelsize=7)

# ── esfera de Bloch ───────────────────────────────────────
b = qt.Bloch(fig=fig)
b.sphere_color = '#111133'
b.sphere_alpha = 0.10
b.frame_color  = '#333355'
b.frame_alpha  = 0.18
b.font_color   = '#aaaacc'
b.font_size    = 10
b.show()
ax3d = b.axes
ax3d.set_facecolor('#0d0d0d')
ax3d.set_position([0.0, 0.04, 0.44, 0.94])

QCOLORS = ['#EF9F27', '#5DCAA5', '#ED93B1']
QLABELS = ['Alice msg', 'Alice EPR', 'Bob EPR']

vectors = []
dots    = []
trails  = []
qlabels_3d = []

for i, (col, lbl) in enumerate(zip(QCOLORS, QLABELS)):
    v, = ax3d.plot([], [], [], linewidth=2.8, color=col, zorder=9)
    d, = ax3d.plot([], [], [], 'o', markersize=9, color=col,
                   zorder=10, markeredgecolor='white', markeredgewidth=0.4)
    tr, = ax3d.plot([], [], [], linewidth=1.0, color=col, alpha=0.3)
    vectors.append(v)
    dots.append(d)
    trails.append(tr)

phase_lbl = ax3d.text2D(0.03, 0.06, '', transform=ax3d.transAxes,
                         fontsize=10, color='white',
                         bbox=dict(facecolor='#1a1a2e', alpha=0.75,
                                   edgecolor='#444466', pad=4))

# legenda manual
for i, (col, lbl) in enumerate(zip(QCOLORS, QLABELS)):
    ax3d.text2D(0.03, 0.96 - i*0.07, lbl,
                transform=ax3d.transAxes,
                color=col, fontsize=8, fontweight='bold')

# ── circuito de teleportacao ──────────────────────────────
ax_circ.axis('off')
ax_circ.set_xlim(0, 10)
ax_circ.set_ylim(-0.5, 3.5)
ax_circ.set_title('Circuito de teleportacao', color='#aaaacc', fontsize=9, pad=4)

# linhas dos qubits
wire_y = [3.0, 2.0, 1.0]
wire_labels = ['Alice: |psi>', 'Alice: |0>', 'Bob:   |0>']
for y, lbl in zip(wire_y, wire_labels):
    ax_circ.plot([0.5, 9.5], [y, y], color='#333355', linewidth=1.0)
    ax_circ.text(0.3, y, lbl, color='#888899', fontsize=7,
                 va='center', ha='right')

# portas do circuito
circ_gates = [
    # (x, wire, label, cor, fase_ativa)
    (2.0, 2.0, 'H',    '#5DCAA5', 'bell'),
    (3.0, 1.5, 'CNOT', '#5DCAA5', 'bell'),
    (4.5, 1.5, 'CNOT', '#EF9F27', 'meas'),
    (5.5, 3.0, 'H',    '#EF9F27', 'meas'),
    (6.5, 3.0, 'M',    '#EF9F27', 'meas'),
    (6.5, 2.0, 'M',    '#EF9F27', 'meas'),
    (8.0, 1.0, 'X',    '#7F77DD', 'correct'),
    (9.0, 1.0, 'Z',    '#7F77DD', 'correct'),
]

gate_patches = []
for (x, y, lbl, col, _) in circ_gates:
    bx = mpatches.FancyBboxPatch(
        (x-0.4, y-0.35), 0.8, 0.7,
        boxstyle='round,pad=0.04',
        facecolor='#1a1a2e', edgecolor=col,
        linewidth=1.2, transform=ax_circ.transData, zorder=3
    )
    ax_circ.add_patch(bx)
    ax_circ.text(x, y, lbl, ha='center', va='center',
                 color=col, fontsize=8, fontweight='bold', zorder=4)
    gate_patches.append((bx, col))

# linhas classicas (tracejadas) — canal classico
ax_circ.plot([6.5, 7.2, 7.2, 8.0], [3.0, 3.0, 1.0, 1.0],
             color='#ED93B1', linewidth=0.8, linestyle='--', alpha=0.4,
             zorder=2)
ax_circ.plot([6.5, 7.5, 7.5, 9.0], [2.0, 2.0, 1.0, 1.0],
             color='#ED93B1', linewidth=0.8, linestyle='--', alpha=0.4,
             zorder=2)

# seta indicadora no circuito
circ_indicator, = ax_circ.plot([], [], 'v', color='#EF9F27',
                                markersize=8, zorder=5)

# ── fidelidade ───────────────────────────────────────────
fid_vals = []
for f in range(N_FRAMES):
    q0, q1, q2 = all_pos[f]
    # fidelidade aproximada pelo produto interno dos vetores de Bloch
    fid = float(np.dot(v_in, q2) + 1) / 2
    fid_vals.append(np.clip(fid, 0, 1))
fid_vals = np.array(fid_vals)

ax_fid.plot(np.arange(N_FRAMES), fid_vals,
            color='#E24B4A', linewidth=1.0, alpha=0.3)
fid_line, = ax_fid.plot([], [], color='#E24B4A', linewidth=2.0)
fid_dot,  = ax_fid.plot([], [], 'o', color='#EF9F27', markersize=5, zorder=5)
ax_fid.set_xlim(0, N_FRAMES)
ax_fid.set_ylim(-0.05, 1.10)
ax_fid.axhline(1.0, color='#5DCAA5', linewidth=0.8, linestyle=':')
style_ax(ax_fid, 'Fidelidade', ylabel='F')
ax_fid.set_xticks([])

# ── probabilidades dos resultados ────────────────────────
result_keys  = list(results.keys())
result_probs = [results[k]['prob'] for k in result_keys]
result_cols  = ['#5DCAA5', '#EF9F27', '#ED93B1', '#7F77DD']
bars_prob = ax_prob.bar(
    [str(k) for k in result_keys], result_probs,
    color=result_cols, edgecolor='#333355', linewidth=0.5
)
ax_prob.set_ylim(0, 0.4)
ax_prob.axhline(0.25, color='#444466', linewidth=0.6, linestyle=':')
style_ax(ax_prob, 'P(resultado)', ylabel='Prob')
ax_prob.tick_params(axis='x', labelsize=7, colors='#888899')

# ── bits classicos ────────────────────────────────────────
ax_bits.axis('off')
ax_bits.set_xlim(0, 1)
ax_bits.set_ylim(0, 1)
ax_bits.set_title('Canal classico', color='#aaaacc', fontsize=9, pad=4)
bit_b0 = ax_bits.text(0.5, 0.65, 'b0 = ?', ha='center', va='center',
                       color='#888899', fontsize=14, fontweight='bold')
bit_b1 = ax_bits.text(0.5, 0.35, 'b1 = ?', ha='center', va='center',
                       color='#888899', fontsize=14, fontweight='bold')
bit_arrow = ax_bits.annotate('', xy=(0.5, 0.05), xytext=(0.5, 0.88),
                              arrowprops=dict(arrowstyle='->', color='#444466',
                                             lw=1.5))

# ── painel info ───────────────────────────────────────────
ax_info.axis('off')
info_header = (
    f"Estado de entrada\n"
    f"{'─'*18}\n"
    f"theta = {THETA:.3f} rad\n"
    f"phi   = {PHI:.3f} rad\n"
    f"{'─'*18}\n"
    f"Bloch:\n"
    f"x={v_in[0]:.2f} y={v_in[1]:.2f}\n"
    f"z={v_in[2]:.2f}"
)
ax_info.text(0.05, 0.98, info_header, transform=ax_info.transAxes,
             color='#aaaacc', fontsize=8, va='top',
             fontfamily='monospace',
             bbox=dict(facecolor='#111133', alpha=0.6,
                       edgecolor='#333355', pad=5))

phase_info = ax_info.text(0.05, 0.42, '', transform=ax_info.transAxes,
                           color='white', fontsize=9,
                           fontweight='bold', va='top')
fid_info   = ax_info.text(0.05, 0.28, '', transform=ax_info.transAxes,
                           color='#E24B4A', fontsize=9, va='top')
bob_info   = ax_info.text(0.05, 0.14, '', transform=ax_info.transAxes,
                           color='#ED93B1', fontsize=9, va='top')

# ── animacao ──────────────────────────────────────────────
# historico de trilha para qubit de Bob
bob_trail_pts = []

def update(frame):
    ph, idx, label, color = frame_list[frame]
    q0, q1, q2 = all_pos[frame]

    # vetores e pontos
    for i, (pos, vec, dot, trail) in enumerate(
            zip([q0, q1, q2], vectors, dots, trails)):
        vec.set_data([0, pos[0]], [0, pos[1]])
        vec.set_3d_properties([0, pos[2]])
        dot.set_data([pos[0]], [pos[1]])
        dot.set_3d_properties([pos[2]])

    # trilha do qubit de Bob
    if ph in ('correct', 'verify', 'classical'):
        bob_trail_pts.append(q2.copy())
    if len(bob_trail_pts) > 1:
        tp = np.array(bob_trail_pts)
        trails[2].set_data(tp[:, 0], tp[:, 1])
        trails[2].set_3d_properties(tp[:, 2])

    # camera
    ax3d.view_init(elev=20, azim=frame * 0.6 + 30)

    # label fase
    phase_lbl.set_text(label)
    phase_lbl.set_color(color)

    # fidelidade acumulada
    fid_line.set_data(np.arange(frame+1), fid_vals[:frame+1])
    fid_dot.set_data([frame], [fid_vals[frame]])

    # bits classicos
    if ph in ('classical', 'correct', 'verify'):
        b0, b1 = best
        bit_b0.set_text(f'b0 = {b0}')
        bit_b1.set_text(f'b1 = {b1}')
        bit_b0.set_color('#EF9F27')
        bit_b1.set_color('#EF9F27')
        bit_arrow.arrow_patch.set_color('#EF9F27')
    elif ph == 'meas':
        bit_b0.set_text('b0 = ?')
        bit_b1.set_text('b1 = ?')
        bit_b0.set_color('#ED93B1')
        bit_b1.set_color('#ED93B1')
        bit_arrow.arrow_patch.set_color('#ED93B1')
    else:
        bit_b0.set_color('#888899')
        bit_b1.set_color('#888899')
        bit_arrow.arrow_patch.set_color('#444466')

    # destaque no circuito
    phase_to_gate_phase = {
        'bell': 'bell', 'meas': 'meas',
        'correct': 'correct', 'verify': 'correct'
    }
    active_phase = phase_to_gate_phase.get(ph, '')
    x_indicator  = {'bell': 2.5, 'meas': 5.5,
                     'correct': 8.5}.get(active_phase, -1)
    if x_indicator > 0:
        circ_indicator.set_data([x_indicator], [3.8])
    else:
        circ_indicator.set_data([], [])

    for (bx, col), (_, _, _, _, gate_ph) in zip(gate_patches, circ_gates):
        if gate_ph == active_phase:
            bx.set_facecolor('#2a2a4e')
            bx.set_linewidth(2.0)
        else:
            bx.set_facecolor('#1a1a2e')
            bx.set_linewidth(1.0)

    # painel info
    phase_info.set_text(label)
    phase_info.set_color(color)
    fid_info.set_text(f'Fidelidade: {fid_vals[frame]:.3f}')
    bob_info.set_text(
        f'Bob: ({q2[0]:.2f},{q2[1]:.2f},{q2[2]:.2f})'
    )

    return []

ani = FuncAnimation(fig, update, frames=N_FRAMES,
                    interval=45, blit=False)

plt.suptitle('Teleportacao Quantica', color='#ccccee',
             fontsize=13, y=0.99)

if SAVE_GIF:
    print(f"Salvando GIF ({N_FRAMES} frames)... aguarde.")
    ani.save('teleportation.gif',
             writer=PillowWriter(fps=GIF_FPS), dpi=110)
    print("Salvo: teleportation.gif")

plt.show()