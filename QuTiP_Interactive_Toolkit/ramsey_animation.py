import sys
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.stdout.reconfigure(encoding='utf-8')

# ════════════════════════════════════════════════════════
#  CONFIGURE AQUI
# ════════════════════════════════════════════════════════
DETUNING      = 0.8    # delta: diferenca entre laser e atomo (Hz)
T_FREE        = 6.0    # tempo de precessao livre
OMEGA_R       = 1.0    # frequencia de Rabi dos pulsos

FRAMES_PULSE1 = 40     # frames do pulso pi/2 inicial
FRAMES_FREE   = 90     # frames da precessao livre
FRAMES_PULSE2 = 40     # frames do pulso pi/2 final
FRAMES_PAUSE  = 20     # pausa entre fases

SAVE_GIF      = True
GIF_FPS       = 25
# ════════════════════════════════════════════════════════

# ── cores por fase ────────────────────────────────────────
COLORS = {
    'pulse1':  '#EF9F27',   # laranja  — primeiro pulso
    'free':    '#5DCAA5',   # verde    — precessao livre
    'pulse2':  '#ED93B1',   # rosa     — segundo pulso
    'pause':   '#444466',   # cinza    — pausa
}

PHASE_LABELS = {
    'pulse1': 'Pulso pi/2',
    'pause1': '...',
    'free':   'Precessao livre',
    'pause2': '...',
    'pulse2': 'Pulso pi/2',
    'pause3': 'Medicao',
}

PHASE_DESC = {
    'pulse1': 'Rotacao de 90 em torno de X\nQubit vai de |0> para |+>',
    'pause1': '',
    'free':   f'Qubit prece em torno de Z\nDetuning delta = {DETUNING:.2f}',
    'pause2': '',
    'pulse2': 'Segundo pulso pi/2\nConverte fase em populacao',
    'pause3': 'Estado final medido em Z\nFranja de Ramsey detectada',
}

# ── Hamiltonianos ─────────────────────────────────────────
# pulso pi/2: rotacao de 90 em torno de X
# tempo de pulso: pi/(2*Omega_R)
T_PULSE = np.pi / (2 * OMEGA_R)

H_pulse = OMEGA_R * qt.sigmax() / 2
H_free  = DETUNING * qt.sigmaz() / 2

# ── calcula trajetorias de cada fase ──────────────────────
def evolve_traj(H, psi0, t_total, n_pts):
    tlist = np.linspace(0, t_total, n_pts)

    res = qt.sesolve(
        H, psi0, tlist,
        e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
        options=qt.Options(store_states=True) 
    )

    traj = np.column_stack([
        np.array(res.expect[0]),
        np.array(res.expect[1]),
        np.array(res.expect[2]),
    ])

    return traj, res.states[-1]

print("Calculando trajetorias...")
psi0 = qt.basis(2, 0)   # inicia em |0>

traj_p1, psi_after_p1 = evolve_traj(H_pulse, psi0,
                                     T_PULSE, FRAMES_PULSE1)
traj_fr, psi_after_fr = evolve_traj(H_free,  psi_after_p1,
                                     T_FREE,  FRAMES_FREE)
traj_p2, psi_final    = evolve_traj(H_pulse, psi_after_fr,
                                     T_PULSE, FRAMES_PULSE2)

# ── lista global de frames ────────────────────────────────
# (fase, idx_local, cor)
frame_list = []
for f in range(FRAMES_PULSE1):
    frame_list.append(('pulse1', f, COLORS['pulse1']))
for f in range(FRAMES_PAUSE):
    frame_list.append(('pause1', f, COLORS['pause']))
for f in range(FRAMES_FREE):
    frame_list.append(('free',   f, COLORS['free']))
for f in range(FRAMES_PAUSE):
    frame_list.append(('pause2', f, COLORS['pause']))
for f in range(FRAMES_PULSE2):
    frame_list.append(('pulse2', f, COLORS['pulse2']))
for f in range(FRAMES_PAUSE):
    frame_list.append(('pause3', f, COLORS['pause']))

N_FRAMES = len(frame_list)

# posicao (x,y,z) em cada frame global
def traj_for(phase, idx):
    if phase == 'pulse1':
        return traj_p1[idx]
    if phase in ('pause1',):
        return traj_p1[-1]
    if phase == 'free':
        return traj_fr[idx]
    if phase in ('pause2',):
        return traj_fr[-1]
    if phase == 'pulse2':
        return traj_p2[idx]
    return traj_p2[-1]   # pause3

all_pts = np.array([traj_for(ph, idx) for ph, idx, _ in frame_list])

# varredura de detuning para franja de Ramsey
print("Calculando franjas de Ramsey...")
detunings  = np.linspace(-3, 3, 200)
P_excited  = []
P_exc_op   = qt.basis(2, 0) * qt.basis(2, 0).dag()
for d in detunings:
    Hf = d * qt.sigmaz() / 2
    _, psi_a = evolve_traj(H_pulse, psi0,       T_PULSE, 20)
    _, psi_b = evolve_traj(Hf,      psi_a,      T_FREE,  20)
    _, psi_c = evolve_traj(H_pulse, psi_b,      T_PULSE, 20)
    P_excited.append(float(qt.expect(P_exc_op, psi_c).real))
P_excited = np.array(P_excited)

# ── figura ────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor('#0d0d0d')

gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[1.6, 1],
                       width_ratios=[1.8, 1.1, 1.1],
                       hspace=0.4, wspace=0.35)

ax_bloch  = fig.add_subplot(gs[:, 0])    # esfera (ocupa 2 linhas)
ax_franja = fig.add_subplot(gs[0, 1])    # franja de Ramsey
ax_circ   = fig.add_subplot(gs[0, 2])    # diagrama do circuito
ax_prob   = fig.add_subplot(gs[1, 1])    # P(excitado) vs tempo
ax_info   = fig.add_subplot(gs[1, 2])    # info

for ax in [ax_franja, ax_circ, ax_prob, ax_info]:
    ax.set_facecolor('#0d0d0d')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_title(title,   color='#aaaacc', fontsize=9,  pad=4)
    ax.set_xlabel(xlabel, color='#888899', fontsize=8)
    ax.set_ylabel(ylabel, color='#888899', fontsize=8)
    ax.tick_params(colors='#888899', labelsize=7)

# ── esfera de Bloch ───────────────────────────────────────
b = qt.Bloch(fig=fig)
b.sphere_color = '#111133'
b.sphere_alpha = 0.10
b.frame_color  = '#333355'
b.frame_alpha  = 0.20
b.font_color   = '#aaaacc'
b.font_size    = 11
b.show()
ax3d = b.axes
ax3d.set_facecolor('#0d0d0d')
ax3d.set_position([0.0, 0.05, 0.50, 0.92])

# artistas da esfera
trail_p1, = ax3d.plot([], [], [], color=COLORS['pulse1'],
                       linewidth=1.5, alpha=0.5)
trail_fr, = ax3d.plot([], [], [], color=COLORS['free'],
                       linewidth=1.5, alpha=0.5)
trail_p2, = ax3d.plot([], [], [], color=COLORS['pulse2'],
                       linewidth=1.5, alpha=0.5)
vector,   = ax3d.plot([], [], [], linewidth=3.0, zorder=9)
dot,      = ax3d.plot([], [], [], 'o', markersize=10, zorder=10,
                       markeredgecolor='white', markeredgewidth=0.5)

phase_lbl = ax3d.text2D(0.03, 0.06, '', transform=ax3d.transAxes,
                         fontsize=10, color='white',
                         bbox=dict(facecolor='#1a1a2e', alpha=0.75,
                                   edgecolor='#444466', pad=4))
state_lbl = ax3d.text2D(0.03, 0.94, '', transform=ax3d.transAxes,
                         fontsize=8, color='#aaaacc')

# ── franja de Ramsey ──────────────────────────────────────
ax_franja.plot(detunings, P_excited, color='#7F77DD',
               linewidth=1.8, alpha=0.6)
franja_dot, = ax_franja.plot([], [], 'o', color='#EF9F27',
                              markersize=7, zorder=5)
ax_franja.axvline(x=DETUNING, color='#444466',
                   linestyle=':', linewidth=0.8)
ax_franja.set_xlim(-3, 3)
ax_franja.set_ylim(-0.05, 1.05)
style_ax(ax_franja, 'Franja de Ramsey',
         xlabel='Detuning delta', ylabel='P(excitado)')

# ── circuito ──────────────────────────────────────────────
ax_circ.axis('off')
ax_circ.set_xlim(0, 10)
ax_circ.set_ylim(0, 4)
ax_circ.set_title('Sequencia', color='#aaaacc', fontsize=9, pad=4)

# linha de qubit
ax_circ.plot([0.5, 9.5], [2, 2], color='#444466', linewidth=1.2)
ax_circ.text(0.2, 2, '|0>', color='#aaaacc', fontsize=9, va='center')

circ_elements = [
    (2.0,  'pi/2', COLORS['pulse1']),
    (5.0,  'T',    COLORS['free']),
    (8.0,  'pi/2', COLORS['pulse2']),
]
circ_boxes = []
for x, lbl, col in circ_elements:
    box = mpatches.FancyBboxPatch(
        (x - 0.6, 1.4), 1.2, 1.2,
        boxstyle='round,pad=0.05',
        facecolor='#1a1a2e', edgecolor=col, linewidth=1.5,
        transform=ax_circ.transData, zorder=2
    )
    ax_circ.add_patch(box)
    ax_circ.text(x, 2, lbl, ha='center', va='center',
                 color=col, fontsize=8, fontweight='bold', zorder=3)
    circ_boxes.append(box)

circ_arrow = ax_circ.annotate(
    '', xy=(1.4, 2), xytext=(0.8, 2),
    arrowprops=dict(arrowstyle='->', color='#EF9F27', lw=1.5)
)

# ── P(excitado) vs tempo ──────────────────────────────────
P_exc_op = qt.basis(2, 0) * qt.basis(2, 0).dag()
pexc_vals = np.array([
    float(qt.expect(P_exc_op, qt.Qobj(all_pts[f])).real)
    if False else
    float(qt.expect(P_exc_op,
        (-1j * (H_pulse if frame_list[f][0] in ('pulse1','pulse2')
                else H_free) * 0).expm() * psi0).real)
    for f in range(N_FRAMES)
])

# recalcula P_exc corretamente frame a frame
pexc_vals = []
psi_cur   = psi0
for ph, idx, _ in frame_list:
    if ph == 'pulse1':
        t   = T_PULSE * idx / max(FRAMES_PULSE1 - 1, 1)
        U   = (-1j * H_pulse * t).expm()
        psi_tmp = U * psi0
    elif ph == 'pause1':
        psi_tmp = psi_after_p1
    elif ph == 'free':
        t   = T_FREE * idx / max(FRAMES_FREE - 1, 1)
        U   = (-1j * H_free * t).expm()
        psi_tmp = U * psi_after_p1
    elif ph == 'pause2':
        psi_tmp = psi_after_fr
    elif ph == 'pulse2':
        t   = T_PULSE * idx / max(FRAMES_PULSE2 - 1, 1)
        U   = (-1j * H_pulse * t).expm()
        psi_tmp = U * psi_after_fr
    else:
        psi_tmp = psi_final
    pexc_vals.append(float(qt.expect(P_exc_op, psi_tmp).real))
pexc_vals = np.array(pexc_vals)

ax_prob.plot(np.arange(N_FRAMES), pexc_vals,
             color='#5DCAA5', linewidth=1.0, alpha=0.3)
prob_line, = ax_prob.plot([], [], color='#5DCAA5', linewidth=1.8)
prob_dot,  = ax_prob.plot([], [], 'o', color='#EF9F27',
                           markersize=5, zorder=5)
ax_prob.set_xlim(0, N_FRAMES)
ax_prob.set_ylim(-0.05, 1.05)

# regioes coloridas por fase
phase_starts = {
    'pulse1': 0,
    'pause1': FRAMES_PULSE1,
    'free':   FRAMES_PULSE1 + FRAMES_PAUSE,
    'pause2': FRAMES_PULSE1 + FRAMES_PAUSE + FRAMES_FREE,
    'pulse2': FRAMES_PULSE1 + FRAMES_PAUSE + FRAMES_FREE + FRAMES_PAUSE,
    'pause3': FRAMES_PULSE1 + FRAMES_PAUSE + FRAMES_FREE + FRAMES_PAUSE + FRAMES_PULSE2,
}
phase_widths = {
    'pulse1': FRAMES_PULSE1, 'pause1': FRAMES_PAUSE,
    'free':   FRAMES_FREE,   'pause2': FRAMES_PAUSE,
    'pulse2': FRAMES_PULSE2, 'pause3': FRAMES_PAUSE,
}
phase_cols = {
    'pulse1': COLORS['pulse1'], 'pause1': '#1a1a1a',
    'free':   COLORS['free'],   'pause2': '#1a1a1a',
    'pulse2': COLORS['pulse2'], 'pause3': '#1a1a1a',
}
for ph, x0 in phase_starts.items():
    ax_prob.axvspan(x0, x0 + phase_widths[ph],
                    alpha=0.08, color=phase_cols[ph])
ax_prob.set_xticks([])
style_ax(ax_prob, 'P(excitado) ao longo do tempo', ylabel='P(|1>)')

# ── painel info ───────────────────────────────────────────
ax_info.axis('off')
info_txt = (
    f"Ramsey\n"
    f"{'─'*20}\n"
    f"Omega_R : {OMEGA_R:.2f}\n"
    f"Delta   : {DETUNING:.2f}\n"
    f"T_free  : {T_FREE:.2f}\n"
    f"{'─'*20}\n"
    f"T_pulso : {T_PULSE:.3f}\n"
)
ax_info.text(0.05, 0.98, info_txt, transform=ax_info.transAxes,
             color='#aaaacc', fontsize=8.5, va='top',
             fontfamily='monospace',
             bbox=dict(facecolor='#111133', alpha=0.6,
                       edgecolor='#333355', pad=5))

phase_info = ax_info.text(0.05, 0.38, '', transform=ax_info.transAxes,
                           color='white', fontsize=9,
                           fontweight='bold', va='top')
desc_info  = ax_info.text(0.05, 0.26, '', transform=ax_info.transAxes,
                           color='#888899', fontsize=8, va='top')
pexc_info  = ax_info.text(0.05, 0.08, '', transform=ax_info.transAxes,
                           color='#5DCAA5', fontsize=9, va='top')

# ── animacao ──────────────────────────────────────────────
# mapeia fase -> indice do box do circuito
PHASE_BOX = {'pulse1': 0, 'pause1': 0,
             'free': 1,   'pause2': 1,
             'pulse2': 2, 'pause3': 2}

def update(frame):
    ph, idx, color = frame_list[frame]
    pt = all_pts[frame]

    # trilhas acumuladas por fase
    p1_end = phase_starts['pause1']
    fr_end = phase_starts['pause2']
    p2_end = phase_starts['pause3']

    if frame >= 0:
        s, e = 0, min(frame + 1, p1_end)
        if e > s:
            trail_p1.set_data(all_pts[s:e, 0], all_pts[s:e, 1])
            trail_p1.set_3d_properties(all_pts[s:e, 2])

    if frame >= phase_starts['free']:
        s, e = phase_starts['free'], min(frame + 1, fr_end)
        if e > s:
            trail_fr.set_data(all_pts[s:e, 0], all_pts[s:e, 1])
            trail_fr.set_3d_properties(all_pts[s:e, 2])

    if frame >= phase_starts['pulse2']:
        s, e = phase_starts['pulse2'], min(frame + 1, p2_end)
        if e > s:
            trail_p2.set_data(all_pts[s:e, 0], all_pts[s:e, 1])
            trail_p2.set_3d_properties(all_pts[s:e, 2])

    # vetor atual
    vector.set_data([0, pt[0]], [0, pt[1]])
    vector.set_3d_properties([0, pt[2]])
    vector.set_color(color)
    dot.set_data([pt[0]], [pt[1]])
    dot.set_3d_properties([pt[2]])
    dot.set_color(color)

    # camera gira suavemente
    ax3d.view_init(elev=20, azim=frame * 0.7)

    # labels esfera
    phase_lbl.set_text(PHASE_LABELS.get(ph, ph))
    phase_lbl.set_color(color)
    state_lbl.set_text(f'<X>={pt[0]:.2f} <Y>={pt[1]:.2f} <Z>={pt[2]:.2f}')

    # P(excitado) acumulado
    prob_line.set_data(np.arange(frame + 1), pexc_vals[:frame + 1])
    prob_dot.set_data([frame], [pexc_vals[frame]])

    # ponto na franja (so na fase de medicao)
    if ph == 'pause3':
        P_cur = float(qt.expect(P_exc_op, psi_final).real)
        franja_dot.set_data([DETUNING], [P_cur])
    else:
        franja_dot.set_data([], [])

    # destaque no circuito
    box_idx = PHASE_BOX.get(ph, -1)
    for i, (box, (_, _, col)) in enumerate(zip(circ_boxes, circ_elements)):
        if i == box_idx and ph not in ('pause1', 'pause2', 'pause3'):
            box.set_facecolor('#2a2a4e')
            box.set_linewidth(2.5)
        else:
            box.set_facecolor('#0d0d1a')
            box.set_linewidth(1.0)

    # seta no circuito
    x_arr = [1.4, 4.4, 4.4, 7.4, 7.4, 9.2][
        ['pulse1','pause1','free','pause2','pulse2','pause3'].index(ph)
    ]
    circ_arrow.xy    = (x_arr, 2)
    circ_arrow.xyann = (x_arr - 0.6, 2)

    # painel info
    phase_info.set_text(PHASE_LABELS.get(ph, ph))
    phase_info.set_color(color)
    desc_info.set_text(PHASE_DESC.get(ph, ''))
    pexc_info.set_text(f'P(excitado) = {pexc_vals[frame]:.3f}')

    return []

ani = FuncAnimation(fig, update, frames=N_FRAMES,
                    interval=40, blit=False)

plt.suptitle('Experimento de Ramsey', color='#ccccee',
             fontsize=13, y=0.99)

if SAVE_GIF:
    print(f"Salvando GIF ({N_FRAMES} frames)... aguarde.")
    ani.save('ramsey_animation.gif',
             writer=PillowWriter(fps=GIF_FPS), dpi=110)
    print("Salvo: ramsey_animation.gif")

plt.show()