# Grid k=4/5 sweep: sensor cost x alpha x epsilon (runs 880-903)

Switching A,B x 3 passes x 1000 episodes, factored tabular SARSA, no detector,
greedy evaluation. Baselines at sensor cost -0.16: run 871 (k=4), 872 (k=5).

Ceiling for an oracle that knows the matrix: 92.49% at cost -0.16 (it watches 3
cells), 98.29% at cost -0.04 (k=4 and 5), 99.00% at cost 0 (k=5).

Result: nothing beats the -0.16 baselines. Best last-pass segment-end accuracy:
k=4 83.80% (cost -0.04, alpha 0.1, eps 0.15) vs 86.80% baseline; k=5 85.55% vs
89.40% baseline. Cost 0 is worst (47.95-83.50%). Full table in the run folders.

Diagnosis (stationary matrix A, k=4, cost 0, alpha 0.05, 6000 episodes): greedy
accuracy 80.75%, and at interior cells only 28% of the watched cells are among the
true top-4 destinations. At cell 12 the learned weights are ~2.7 for cells 8, 19,
18, 2, 23, 5 -- several unreachable in one step -- while cell 17 (p=0.40) is not in
the top 8.

Cause: the factored update applies the same TD error to every active cell
(step alpha/n_active), so cells switched on together receive identical credit
regardless of which one found the object. Only epsilon-random actions separate
them. Lowering the sensor cost makes it worse: the agent always fills all k
slots, so co-activation is constant and the per-cell signal vanishes.
