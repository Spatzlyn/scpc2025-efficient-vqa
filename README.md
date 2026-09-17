# Efficient Visual Question Answering under a 3B Parameter Budget

1st Place · Samsung Collegiate Programming Challenge (SCPC) 2025, AI Challenge

Yookyung Youn · Korea University · Individual participant

[Technical report](docs/technical-report.md) · [Report PDF](docs/technical-report.pdf) · [Results](results/README.md) · [Official award announcement](https://research.samsung.com/news/Samsung-Electronics-Unveils-Winners-of-11th-Collegiate-Programming-Challenge-as-Part-of-AI-Talent-Discovery-Initiative)

## Overview

This project addresses multiple-choice visual question answering under a strict inference budget of fewer than 3 billion parameters. Starting from the task and competition constraints, I independently developed the solution: selecting the model and training data, designing the compression strategy, implementing training and inference, evaluating alternatives, and preparing the final submission.

As an undergraduate working with personally funded, on-demand GPUs, I concentrated the development into approximately two to three weeks around end-of-semester commitments. I selected experiments by what they could resolve about the next design decision: which components tolerated intervention, which removal combinations preserved task behavior, and when additional training stopped transferring to the competition setting. The [technical report](docs/technical-report.md) reconstructs that process in detail.

I selected InstructBLIP-Flan-T5-XL and combined neuron-level structured pruning of decoder feed-forward networks (FFNs), head-level structured pruning of decoder attention, teacher-student knowledge distillation, and LoRA fine-tuning. The final model contains approximately 2.987B parameters, a 25.8% reduction from the 4.023B original. Its reported public leaderboard weighted accuracy is 86.59%.

This repository presents the technical report, newly drawn figures, and aggregate experimental results.

## Design Rationale

The final pipeline emerged through revisions to the initial plan. The report follows the questions, hypotheses, observations, and decisions behind those revisions.

| Question | Reasoning and resulting decision |
|---|---|
| Which model could I compress and meaningfully investigate? | InstructBLIP's component structure let me examine interventions separately. I accepted a larger initial parameter count to preserve existing alignment and make the compression problem more tractable. |
| Where should I spend the compression budget? | I hypothesized that upstream representation damage and disruption of the visual-language interface could be costly. Recalled encoder trials showed severe losses, directing further investigation toward decoder components. |
| Was low activation enough to identify expendable computation? | Irregular masking results challenged that assumption. I refined activation clusters and evaluated removal combinations conditionally, using a greedy search rather than trusting magnitude alone. |
| Which explored interventions still justified their cost? | Hidden-state masking offered limited extra benefit once FFN compression supplied substantial savings. I excluded it from the final structure and used FFN neuron and attention-head removal. |
| How should the pruned model recover? | I treated KD as stabilization and selected the original unpruned model as teacher. The goal was to restore useful behavior in the compressed architecture before further adaptation. |
| How could the remaining parameter allowance improve performance? | I implemented LoRA wrappers for the changed projection dimensions, froze the backbone, and mixed previously correct and incorrect examples to balance preservation and correction. |
| What did a strong final score fail to capture? | I observed poorer language generation despite retained answer-selection ability. This qualitative limitation motivated my interest in evaluation beyond task accuracy and in trustworthy AI. |

The rationale is reconstructed from my recollection alongside archived implementation and results. The report labels qualitative comparisons whose complete numerical records have not been recovered. It also distinguishes the methods used during the competition from literature consulted afterward, including my adaptation of Wanda's activation-and-weight principle. The contribution is the independently developed solution and its experimental decision process.

## Final Method

![Solution design: model selection, structured pruning, teacher-guided stabilization, and balanced-data LoRA fine-tuning](assets/method-overview.png)

FFN pruning removes intermediate neurons and their associated input/output weights, reducing the FFN intermediate widths. Attention pruning removes whole heads from decoder self-attention and cross-attention, reducing the internal projection widths. Both operations rebuild smaller dense layers. The FFN importance score is inspired by Wanda's use of weights and activations; this implementation prunes neuron groups rather than individual weight elements. The earlier neuron-masking experiments evaluated candidate removals before structural compaction.

The FFN criterion combines mean absolute gated activation with the L2 norm of the output-projection column. The selected 95% removal applies to decoder FFN neurons in aggregate, with different retained widths per layer. Head candidates use cosine similarity of original-model attention maps and task evaluation. The final system does not include the exploratory hidden-state masking.

Teacher-student stabilization uses output-distribution supervision and target answers. Custom query/value LoRA adapters then add 8,159,232 parameters, with the backbone frozen and a 1:1 mixture of previously correct and incorrect A-OKVQA training examples. The [report](docs/technical-report.md) provides the implementation details and distinguishes retained settings from exploratory trials.

## Results

| Stage | Parameters | A-OKVQA validation accuracy (1,000 examples) | Public weighted accuracy |
|---|---:|---:|---:|
| Original InstructBLIP-Flan-T5-XL | 4,022,969,088 | 79.00% | — |
| Structured pruning | 2,978,586,976 | 76.60% | 82.83% |
| Knowledge distillation | Same pruned architecture | — | 85.73% |
| LoRA fine-tuning: final submission | 2,986,746,208 | — | 86.59% |

Pruning reduced local A-OKVQA validation accuracy by 2.40 percentage points (79.00% to 76.60%). The validation and public leaderboard columns refer to different evaluation settings and must be compared within their own columns. A dash means the consulted records do not supply a value for that particular metric and stage; the original model's validation accuracy is reported above.

The final solution improves the reported score by 3.76 percentage points over the pruned stage while remaining below 3B parameters, including the added adapters. These are historical competition results, not new measurements for this documentation release. The sequence describes cumulative stages; it does not isolate each component's causal effect.

![Parameter counts and public leaderboard scores, plotted separately](assets/results-summary.png)

The first-place award and the public leaderboard score are distinct outcomes. Samsung's [official announcement](https://research.samsung.com/news/Samsung-Electronics-Unveils-Winners-of-11th-Collegiate-Programming-Challenge-as-Part-of-AI-Talent-Discovery-Initiative) documents the award. The [competition rules](https://www.dacon.io/en/competitions/official/236500/overview/rules) describe the evaluation and inference budget.

## Repository Guide

| Path | Contents |
|---|---|
| [docs/technical-report.md](docs/technical-report.md) | Detailed decision process, hypotheses, design revisions, implementation, evidence, limitations, and references |
| [docs/technical-report.pdf](docs/technical-report.pdf) | Downloadable version of the report |
| [assets/](assets/) | Original overview diagram and result plots in PNG and SVG |
| [results/](results/) | Aggregate CSV tables and notes on units, provenance, and interpretation |

## Materials and Scope

The report documents the historical 2025 solution within the model-eligibility and compute constraints of that period. Its references distinguish existing methods, retrospectively connected work available before the competition, and research published afterward. Independent problem solving is not presented as a claim to have originated those methods.

The public release documents the solution and its reported evidence. Implementation notebooks, model weights, raw datasets, and competition presentation files are not included. This release therefore does not provide an executable reproduction package. The figures were newly drawn for this repository; they are not slide screenshots.

The project builds on [InstructBLIP](https://arxiv.org/abs/2305.06500), [A-OKVQA](https://github.com/allenai/aokvqa), [Wanda](https://arxiv.org/abs/2306.11695), [knowledge distillation](https://arxiv.org/abs/1503.02531), and [LoRA](https://arxiv.org/abs/2106.09685). References and the distinction between existing methods and project-specific decisions appear in the report.
