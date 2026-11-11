"""Reinforcement-learning core shared by the visual-servoing (Sec. IV) and attitude (Sec. V) controllers.

utility.py  - constrained utility and tanh policy, eqs. (11)-(13), (15), (18), (25)-(26) (and (34)-(36), (42))
basis.py    - polynomial basis functions for critic/actor, eqs. (23)-(24) (and (40)-(41)); choice is [Eng] (A16)
data.py     - off-policy data collection and windowing, Alg. 3/4 Step 1 (A17, A18)
offpolicy_irl.py - Bellman rows (27)-(28), least squares (29), Alg. 3 Steps 2-4 (and (39)-(41), Alg. 4)
"""
