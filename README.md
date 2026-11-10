# vsfc — reproduction of input-constrained visual servoing formation control

Pure-Python reproduction of X. Yi, H. Liu, H. Duan, J. Qiu, "Input-Constrained Visual Servoing Formation Control
for Quadrotors Using Off-Policy Reinforcement Learning," IEEE Trans. Cybernetics 56(6):3540–3550, 2026,
DOI 10.1109/TCYB.2026.3656290. M.Tech thesis project.

The paper PDF is not included (IEEE copyright); obtain it via the DOI and place it at `papers/base_paper.pdf`.

- Equation → code map and all ambiguities: `docs/equation_map.md`
- Decisions, status, error log: `docs/RESEARCH_LOG.md`
- Case-study parameters (with sources): `config/paper_case_study.yaml`

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest
```
