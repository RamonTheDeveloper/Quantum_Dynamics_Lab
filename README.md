# ⚛️ Quantum Dynamics Lab

<img width="720" height="720" alt="bloch_animation" src="https://github.com/user-attachments/assets/565ff810-688d-44e9-8349-d31c29e3fed3" />
<img width="560" height="560" alt="multi_qubit_bloch" src="https://github.com/user-attachments/assets/c2d7041e-f318-41f8-8b02-d7f6d63a6113" />
<img width="400" height="350" alt="gates_animation" src="https://github.com/user-attachments/assets/8e95e342-0994-49e8-916e-ac9803368103" />

Bem-vindo ao **Quantum Dynamics Lab**! Este repositório é um ecossistema de simulação voltado para a **Física Computacional** e **Informação Quântica**. O foco principal é a visualização interativa da dinâmica de sistemas de qubits e a implementação de protocolos quânticos fundamentais.

Diferente de simuladores puramente abstratos, este laboratório utiliza o **QuTiP** para resolver a Equação de Schrödinger e simular Hamiltonianos reais, permitindo a observação de fenômenos como Oscilações de Rabi, Interferometria de Ramsey e Tomografia de Estado.

## 🚀 Estrutura do Projeto e Scripts

O laboratório está organizado em módulos que cobrem desde o controle de um único qubit até protocolos de comunicação quântica:

### 1. Visualização e Dinâmica (Bloch Sphere Engine)
* **`simulator_bloch_sphere.py`**: Simulador interativo via CLI. Via terminal interativo, escolha estados iniciais e aplique sequências de portas ($H, X, Y, Z, S, T, RX90, RY90$) para ver o resultado na esfera de Bloch.
* **`bloch_sphere_animation.py`** & **`qubit_trajetory.py`**: Scripts focados na evolução temporal sob um Hamiltoniano específico, traçando a trajetória contínua do vetor de estado.
* **`bloch_sphere_gates_animation.py`**: Gera animações fluidas de transformações unitárias, ideal para entender como cada porta rotaciona o estado.
* **`multi_qubit_bloch.py`**: Visualização simultânea de 3 qubits independentes sob diferentes Hamiltonianos, comparando evoluções temporais em paralelo.

### 2. Fenômenos de Ressonância e Interferometria
* **`rabi_eletromagnetico.py`** & **`rabi_oscilations.py`**: Simulações das Oscilações de Rabi. Analisa a probabilidade de transição entre estados fundamental e excitado sob a influência de campos externos e diferentes *detunings*.
* **`ramsey_animation.py`**: Uma simulação completa da técnica de Franjas de Ramsey, essencial para metrologia quântica e medição de tempos de coerência.

### 3. Protocolos Avançados e Tomografia
* **`teleportation_quantum.py`**: Implementação visual do protocolo de Teletransporte Quântico. O script anima cada fase: criação do par de Bell, medição conjunta e correção clássica.
* **`pauli_setup_quantum.py`**: Implementação de **Tomografia de Estado Quântico**. Simula medições projetivas reais com distribuição binomial (shots) para reconstruir a matriz densidade do estado.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Simulação Quântica:** [QuTiP](https://qutip.org/) (Quantum Toolbox in Python)
* **Cálculo Numérico:** NumPy
* **Renderização Científica:** Matplotlib (FuncAnimation, GridSpec, 3D Engine)

## 🎮 Como Executar

AVISO: Será necessário que você tenha PC/Notebook com performance mediana/ou máxima para poder executar, caso contrário, a causa disso será travamentos e fechamento do software automático, isso pq a biblioteca Matplotlib renderiza tudo na CPU (e em Single-Thread).

Todo o cálculo matemático das matrizes do QuTiP e a renderização 3D dos vetores rodando frame por frame acontecem no processador (CPU) em uma única linha de execução. É muita informação visual para o Python desenhar ao vivo, o que gera travamentos e faz a tela congelar.

1. Clone o repositório:
   ```bash
   git clone [https://github.com/RamonTheDeveloper/Quantum_Dynamics_Lab.git](https://github.com/RamonTheDeveloper/Quantum_Dynamics_Lab.git)
2. Instale as bibliotecas para rodar o programa, via Terminal no PyCharm, Visual Studio Code ou Jupyter Notebook:
   pip install qutip numpy matplotlib
3. Execute qualquer um dos simuladores (exemplo):
   python teleportation_quantum.py (ou simplesmente clica em um símbolo de executar que fica no topo)

📊 Aplicações em P&D

Este projeto foi desenvolvido como parte de uma pesquisa em Física Computacional, visando:

• Visualização de sistemas físicos (Spins, Fótons e Átomos de dois níveis).
• Criação de material didático para cursos de graduação em Física e Engenharia.
• Iniciação Científica para o PIBITI/UFAM

Autor: Gabriel Ramon de Souza Ramos
Estudante de Bacharelado em Física - Universidade Federal do Amazonas (UFAM)
