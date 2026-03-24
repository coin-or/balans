# ParBalans Experiments
In ParBalans (ArXiv'25), we found some configs to work well experimentally on Distributional-MIPLib-Hard and MIPLib-Hard.

## 🥇 #1 — Config ID 0 (Best Overall + Most Instance Wins)
File: overall_r1_miplibhard_r1_18wins_HC_SM_id_000.json
Overall rank: #1 (mean gap 6.21%)
miplib-hard rank: #1 (mean gap 15.10%)
Instance wins: 18 (no other config exceeds 7)
Per-domain: Best on srpn (0.00% gap — solves all perfectly)
Strategy: HillClimbing + Softmax(τ=1.36), scores [8,4,2,1], 10 operators

Why: Dominant overall. Wins 22% of all instances outright — 2.5× more than any other config. Heavy on Local_Branching (4 variants) which is the single most statistically significant factor across all datasets (p=0.017). Its conservative HillClimbing acceptance avoids worsening moves, and the steep [8,4,2,1] score vector strongly rewards improvement.
 
## 🥈 #2 — Config ID 105 (Best Cross-Dataset Generalist)
File: overall_r4_miplibhard_r3_real_r1_crossdataset_2wins_SA_EG_id_105.json
Overall rank: #4 (mean gap 6.48%)
real rank: #1 (mean gap 0.58%)
miplib-hard rank: #3 (mean gap 16.27%)
Cross-dataset: Only config in top-10 of 2+ individual benchmarks
Per-domain: Best on ca (0.97% gap)
Strategy: SimulatedAnnealing(step=0.82) + EpsilonGreedy(ε=0.46), scores [5,2,1,0], 12 operators

Why: The only config that consistently ranks high across different benchmark types (real-world + MIPLIB-hard). Its broader 12-operator portfolio (including 3× Rins, 2× Proximity, 3× Local_Branching) gives it flexibility. SimulatedAnnealing acceptance lets it escape local optima, while the moderate ε=0.46 balances exploration vs. exploitation. Best pick when you don't know what problem type you'll face.
 
## 🥉 #3 — Config ID 93 (Simplest + Most Robust)
File: overall_r7_miplibhard_r8_HC_TS_id_093.json
Overall rank: #7 (mean gap 6.91%)
miplib-hard rank: #8 (mean gap 18.25%)
Per-domain: #2 on sc (1.90% gap)
Strategy: HillClimbing + ThompsonSampling, scores [1,1,0,0], only 6 operators

Why: This is the "less is more" config. With only 6 operators — one from each family (Crossover, Mutation_50, Rins_30, Rens_10, Proximity_020, Local_Branching_50) — it's the simplest config in the overall top-10, yet ranks #7 overall. ThompsonSampling with binary [1,1,0,0] scores provides principled Bayesian exploration with minimal tuning knobs (no hyperparameters beyond seed). It's the config to use when you want something parameter-free, fast per iteration, and still competitive.
 
# Summary table:
#1 0: You want the best performance overall 
HillClimbing Softmax 10 
#2 105: You face unknown/diverse problem types
SimulatedAnnealing  EpsilonGreedy 12
#3 93: You want simplicity + no hyperparameters
HillClimbing ThompsonSampling  6

## Recommendation: Config 55 
Dimension | Config 55 | Config 105 | Config 0
Overall rank (combined) | #3 (6.40%) | #4 (6.48%) | #1 (6.21%) |
Avg rank across 3 datasets  | #1 (20.7) — best consistency | Not in top-10  | Not in top-10  |
Per-dataset ranks (miplibd-h / real / miplib-h)  | 32 / 25 / 5  | ~60+ / 1 / 3  | ~23+ / ~65+ / 1  |
Cross-dataset top-10 presence  | 1 dataset + overall  | 2 datasets + overall | 1 dataset + overall
Learning policy  | EpsilonGreedy | EpsilonGreedy  |  Softmax  |
Acceptance  | SimulatedAnnealing  | SimulatedAnnealing  | HillClimbing
Scores  | [5, 4, 2, 0]  | [5, 2, 1, 0]  |  [8, 4, 2, 1]
n_arms  | 15  | 12  | 10

## Config 55 wins as the default because:
* Best cross-dataset consistency — avg rank 20.7, the best of all 180 configs. It never drops below rank 32 on any single dataset.
* Top-5 on the hardest benchmark — rank #5 on miplib-h (hard instances).
* Top-3 overall — rank #3 when all datasets are combined.
* Full operator portfolio — 15 arms covering all operator families (matching the factor analysis that says including all families is universally beneficial).
* Config 0 and Config 105 are "polarizing" — they dominate on 1–2 datasets but perform poorly on others, which is bad for a default.

# Note on Config 105 (runner-up):
For real-world or hard MIPLIB instances,
Config 105 (scores=[5,2,1,0], 12 arms) may outperform Config 55 on those specific benchmarks (rank #1 on real, #3 on miplib-h).
However, its exact 12-operator subset would need to be verified from the Balans_Experiments_final.xlsx arm columns.

