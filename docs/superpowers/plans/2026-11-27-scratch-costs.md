# From-scratch torque/work comparison

Goal: replace fine-tuning evidence with independent PPO initialization and split torque/work penalties, randomized planar hoops, >=3 seeds.

User scope: no main commits; close old PR; retrain from zero. Existing physics/scoring/re-hit rules remain.

Decision: factorial ablation (none, torque, work, both) rather than only baseline/both, to separate effects. 42/43/44 seeds, 1,000,000 requested steps each (1,000,448 actual). Torque penalty 0.5*C, work penalty 0.02*W_abs, time 0.1*dt. These are fixed initial choices, not claimed optimal. All runs get fresh PPO, optimizer, VecNormalize. Save initial parameter hash, normalization count and timestep zero. Same seed must give same initial parameters across conditions; different seeds must differ.

Evaluation: freeze 300 random x/z targets, x [1,1.5], z [.75,1.1], y=0, before training. Use final checkpoints only. Report each seed plus mean/std across 3 training seeds; no pooling 900 episodes as independent training replications. Report success and per-seed common-success costs, null if no common success. Record failure, never silently omit a run. Old Notion results retained as historical fine-tuning evidence.

Tasks:
- [x] RED/GREEN independent work/torque reward deltas and validation tests.
- [x] Add CLI overrides and fresh-start metadata; extend real train/eval test.
- [x] Add reproducible factorial runner with predeclared protocol and completion checks.
- [ ] Run tests, commit code, launch 12 fresh runs; evaluate and summarize all.
- [ ] Review, new PR, Notion result update and labeled GIFs.

Review: independent read-only reviewer found no concrete bug. 38 tests pass; 12-run short end-to-end study passed fresh initialization audit. Fixed checkpoint paths use matching normalization. Main study additionally evaluates 100k/250k/500k/750k/final, all from training step zero.
