# Results and Interpretation

These tables transcribe aggregate numerical results from Yookyung Youn's archived 2025 competition records. No new training or inference was run to create this documentation release. The original presentation and private working files are not distributed. See the [technical report](../docs/technical-report.md) for methods, development-subset sizes, and limitations.

## Files

| File | Contents |
|---|---|
| [stage-results.csv](stage-results.csv) | Whole-model parameter counts, public leaderboard weighted accuracy, and the separately reported A-OKVQA comparison |
| [component-parameters.csv](component-parameters.csv) | Recorded decoder FFN and decoder attention counts before and after pruning |

## Pruning Terminology

The `structured_pruning` stage combines two operations: neuron-level structured pruning of decoder FFNs and head-level structured pruning of decoder self-attention and cross-attention. Both remove groups of connected weights and rebuild smaller dense matrices.

FFN neurons are ranked globally across decoder layers using a Wanda-inspired score: mean absolute gated activation multiplied by the L2 norm of the neuron's output-projection column. The 95% removal rate applies to the aggregate decoder FFN neurons and their associated parameters; retained widths vary by layer. Earlier column-masking experiments evaluate neuron removal without changing matrix dimensions. The final structural implementation removes the corresponding rows of both FFN input projections and columns of the output projection. Attention pruning removes whole heads and their associated projection dimensions.

## Units and Missing Values

- Parameter counts are integer numbers of parameters; 1B means 1,000,000,000 parameters.
- Accuracy columns are percentages, not fractions. For example, 86.59 means 86.59%.
- Blank cells mean no numerical measurement is supplied here. They do not mean zero.
- The KD stage retains the pruned architecture; a separately measured post-KD parameter count is not available in these tables.
- The final parameter count includes the LoRA adapters. The teacher is used for training and is not included in final inference.
- The component table describes selected decoder components, not an exhaustive accounting of every model parameter or every structural adjustment.

## Derived Quantities

The pre-pruning baseline was evaluated: its A-OKVQA validation accuracy was 79.00% on 1,000 examples. The corresponding pruned-model accuracy was 76.60%, a decrease of 2.40 percentage points. The blank public-score cell for the original model does not mean that its validation accuracy was unreported.

| Quantity | Calculation | Rounded value |
|---|---|---:|
| Pruned-model parameter reduction | 100 × (4,022,969,088 − 2,978,586,976) / 4,022,969,088 | 25.96% |
| Final-model parameter reduction | 100 × (4,022,969,088 − 2,986,746,208) / 4,022,969,088 | 25.76% |
| Added adapter parameters | 2,986,746,208 − 2,978,586,976 | 8,159,232 |
| Final margin below 3B | 3,000,000,000 − 2,986,746,208 | 13,253,792 |
| Local validation accuracy change after pruning | 76.60 − 79.00 | −2.40 percentage points |
| Score gain after KD | 85.73 − 82.83 | 2.90 percentage points |
| Further score gain after LoRA | 86.59 − 85.73 | 0.86 percentage points |
| Overall gain over pruning | 86.59 − 82.83 | 3.76 percentage points |

The A-OKVQA accuracies were reported on a 1,000-example validation subset. They are local, unweighted development results and must not be mixed with the competition's public weighted accuracy. No public competition score is assigned to the original, over-budget model.

These are cumulative stages, not isolated ablations. No confidence intervals, repeated-seed statistics, or inference-speed improvements are claimed. The final award is independently documented in [Samsung's announcement](https://research.samsung.com/news/Samsung-Electronics-Unveils-Winners-of-11th-Collegiate-Programming-Challenge-as-Part-of-AI-Talent-Discovery-Initiative); it is distinct from the public leaderboard score.
