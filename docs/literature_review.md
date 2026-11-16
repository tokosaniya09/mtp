# Literature review — working file

Started 2026-11-16 (D37). Feeds the one-page gap statement and hypothesis (D6), then Phase 2.
Rule 6 applies: a paper is cited in the thesis only when its row says **Verified** (DOI, venue, year checked on
the publisher page or dblp) AND one of us has **Read** it. Search-engine summaries are not evidence.

## 1. Review question

> When a follower's visual measurement of its leader is intermittent (occlusion, leader outside the field of
> view, detector failure), how do existing methods keep the leader-state estimate and the formation stable,
> and what do they guarantee?

Sub-questions each paper is read against:

1. **Measurement loss:** is it modelled at all? As what (FOV exit, occlusion, random dropout, bounded-length gap)?
2. **During the gap:** what does the estimator/controller do (hold, open-loop prediction, other sensor, stop)?
3. **Guarantee:** is there a proof for the gap (bounded error, dwell-time condition, ultimate bound), or only
   for continuous measurement?
4. **Sensors and communication:** camera only? UWB? IMU? Is any inter-vehicle communication needed?
5. **Leader motion:** constant velocity, bounded acceleration, known model, learned model?
6. **Validation:** simulation, experiment; how long are the gaps tested?

## 2. Threads

| Thread | Topic | Why we need it | Lead |
|---|---|---|---|
| A | Vision-based (IBVS / bearing) formation and leader-follower control of UAVs; what they assume about visibility | Shows whether our gap (Limitation 8 of the base paper) is already filled | User (reads), Claude (searches) |
| B | Estimation with intermittent measurements: predictor/observer switching, dwell-time conditions, Kalman filtering with missing observations | The theory Phase 2 will most likely build on (TCYB expects a stability result, D36) | Both |
| C | Vision–UWB–IMU fusion for relative localization; how "communication-free" work treats UWB | Decides the open UWB question (D36) | Claude (searches), user (decides) |
| D | Keeping the leader visible: FOV constraints, control barrier functions, re-acquisition / search | Alternative to "survive the gap" is "avoid the gap"; we must position against it | Claude |

## 3. Search log

| Date | Who | Tool | Query | Kept |
|---|---|---|---|---|
| 2026-11-16 | Claude | web search | vision-based leader-follower formation control quadrotor occlusion intermittent visual measurement observer | A3, A4, A5, A6, D1 |
| 2026-11-16 | Claude | web search | image-based visual servoing target tracking intermittent feedback dwell-time predictor observer switched systems | B1, B2 |
| 2026-11-16 | Claude | web search | vision UWB IMU fusion relative localization multi-UAV leader follower formation without communication | C1, C2, A2 |
| 2026-11-16 | Claude | web search | field-of-view constrained visual servoing formation control quadrotors leader loss recovery communication-free | D1, D2 |

## 4. Paper table

Status columns: **Verified** = bibliographic data checked (where); **Read** = who read it and to which pass
(1 = abstract/intro/conclusion/figures, 2 = full read without proofs, 3 = proofs and equations).
Rows marked *candidate* came from a search summary only: details may be wrong until verified.

### Thread A — vision-based formation

| ID | Paper | Verified | Read | Relevance (one line, after reading) |
|---|---|---|---|---|
| A0 | Yi, Liu, Duan, Qiu, "Input-Constrained Visual Servoing Formation Control for Quadrotors Using Off-Policy Reinforcement Learning," IEEE Trans. Cybern. 56(6):3540–3550, 2026, DOI 10.1109/TCYB.2026.3656290 | Yes (PDF) | Both, pass 3 | Base paper; assumes continuous visual measurement |
| A1 | Base paper's own references on vision-based formation (list to be extracted from its Introduction) | — | — | **User task 1** |
| A2 | *candidate:* Evangeliou et al., Frontiers in Robotics and AI, 2022 (published online 2021), DOI 10.3389/frobt.2021.777535 — marker-based leader-follower UAV formation without RF communication; reports tolerance to missed frames. Title to confirm | DOI from URL only | — | |
| A3 | *candidate:* "Vision-Based Formation Control of Quadrotors Using a Bearing-Only Approach," Robotics (MDPI) 13(8):115, 2024. Authors and DOI to confirm | No | — | |
| A4 | *candidate:* RL-based IBVS formation paper with a "leader observer" for GNSS-denied quadrotors without inter-vehicle communication (likely same group as the base paper). Title, authors, venue to identify | No | — | |
| A5 | *candidate:* NTU paper, leader-follower wheeled robots, sliding-window Bayesian estimator fusing odometry with sparse vision when detection drops out. Full reference to identify | No | — | |
| A6 | *candidate:* IEEE TCST 2020, nonholonomic robots, stereo camera, particle filter for intermittently available leader observations. Full reference to identify | No | — | |

### Thread B — estimation with intermittent measurements

| ID | Paper | Verified | Read | Relevance |
|---|---|---|---|---|
| B1 | A. Parikh, T.-H. Cheng, R. Licitra, W. E. Dixon, "A Switched Systems Approach to Image-Based Localization of Targets That Temporarily Leave the Camera Field of View," IEEE Trans. Control Syst. Technol. 26(6):2149–2156, Nov. 2018, DOI 10.1109/TCST.2017.2753163 | dblp + repository record; confirm author list on IEEE Xplore | — | **User task 2.** Dwell-time conditions: minimum visible time, maximum invisible time |
| B2 | A. Parikh, R. Kamalapurkar, W. E. Dixon, "Target Tracking in the Presence of Intermittent Measurements via Motion Model Learning," IEEE Trans. Robot., 2018, DOI 10.1109/TRO.2018.2821169 | dblp entry exists; volume/pages and DOI to confirm on IEEE Xplore | — | Learns the target's motion model while visible, predicts with it while not |
| B3 | B. Sinopoli, L. Schenato, M. Franceschetti, K. Poolla, M. I. Jordan, S. S. Sastry, "Kalman Filtering With Intermittent Observations," IEEE Trans. Autom. Control 49(9):1453–1464, Sept. 2004, DOI 10.1109/TAC.2004.834121 | Yes (dblp, author-hosted PDF) | — | Classic result: critical arrival rate for bounded error covariance. Our Kalman baseline's theory |

### Thread C — vision–UWB–IMU fusion, communication

| ID | Paper | Verified | Read | Relevance |
|---|---|---|---|---|
| C1 | H. Xu, L. Wang, Y. Zhang, K. Qiu, S. Shen, "Decentralized Visual-Inertial-UWB Fusion for Relative State Estimation of Aerial Swarm," IEEE ICRA 2020, pp. 8776–8782, DOI 10.1109/ICRA40945.2020.9196944 (arXiv 2003.05138) | Yes (arXiv listing) | — | Check: does it exchange data between UAVs? |
| C2 | *candidate:* K. Guo, X. Li, L. Xie, "Ultra-Wideband and Odometry-Based Cooperative Relative Localization With Application to Multi-UAV Formation Control," IEEE Trans. Cybern., 2020. DOI, volume, pages to confirm | No | — | Uses UWB for ranging AND communication: evidence for the D36 question |

### Thread D — keeping the leader visible, re-acquisition

| ID | Paper | Verified | Read | Relevance |
|---|---|---|---|---|
| D1 | I. R. Santjoko, R. R. Suganda, M. Pan, B. Hu, "Distributed 3D Leader–Follower Formation Control with Field-of-View Safety via Control Barrier Functions," arXiv 2605.17533, May 2026 (preprint, not peer-reviewed as far as we know) | arXiv listing | — | Avoids FOV loss by a CBF filter; does not handle occlusion by obstacles (to check) |
| D2 | *candidate:* D. Dias, P. U. Lima, A. Martinoli, limited-FOV quadrotor formation control without communication, AAMAS 2016. Title, pages to confirm | No | — | |

## 5. Reading note template (one per paper, appended to section 6)

```
### <ID> <first author year, short title>
Read by / date / pass:
1. Problem in one sentence:
2. Sensors; communication needed? (yes/no, what is exchanged):
3. Leader/target motion assumption:
4. Is measurement loss modelled? How?
5. What happens during the gap (estimator, controller)?
6. Main theoretical result and its conditions (theorem number, in words):
7. Validation (sim/experiment), longest gap tested:
8. What it does NOT do that we need:
9. One thing we could reuse (idea, proof technique, baseline, metric):
10. Things I did not understand (list them; these are questions for the next session):
```

## 6. Reading notes

(empty)

## 7. Emerging gap statement (rewritten as evidence comes in)

Working hypothesis only, not yet supported by the review: no existing vision-based quadrotor formation
controller in the communication-free, GNSS-free setting of the base paper gives a bounded-error guarantee for
the leader-state estimate across measurement gaps. To be confirmed or killed by Threads A and B.
