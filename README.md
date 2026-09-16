# Efficient Visual Question Answering under a 3B Parameter Budget

1st Place · Samsung Collegiate Programming Challenge (SCPC) 2025, AI Challenge

Yookyung Youn · Korea University · Individual participant

[Technical report](docs/technical-report.md) · [Report PDF](docs/technical-report.pdf) · [Results](results/README.md) · [Official award announcement](https://research.samsung.com/news/Samsung-Electronics-Unveils-Winners-of-11th-Collegiate-Programming-Challenge-as-Part-of-AI-Talent-Discovery-Initiative)

## Overview

This project addresses multiple-choice visual question answering under a strict inference budget of fewer than 3 billion parameters. Starting from the task and competition constraints, I independently developed the solution: selecting the model and training data, designing the compression strategy, implementing training and inference, evaluating alternatives, and preparing the final submission.

I selected InstructBLIP-Flan-T5-XL and combined neuron-level structured pruning of decoder feed-forward networks (FFNs), head-level structured pruning of decoder attention, teacher-student knowledge distillation, and LoRA fine-tuning. The final model contains approximately 2.987B parameters, a 25.8% reduction from the 4.023B original. Its reported public leaderboard weighted accuracy is 86.59%.

This repository presents the technical report, newly drawn figures, and aggregate experimental results.

## Approach

![Solution design: model selection, structured pruning, teacher-guided stabilization, and balanced-data LoRA fine-tuning](assets/method-overview.png)

| Decision | Implementation and supporting investigation |
|---|---|
| Preserve a capable pretrained starting point | Select an instruction-following vision-language model and compress its decoder. |
| Allocate compression across components | Rank FFN neurons with a Wanda-inspired activation/weight score and select whole attention heads using similarity analysis; assess removal choices through masking and approximately 60 head-removal configurations. |
| Improve the compressed model | Distill from the original model, then fine-tune query/value adapters using a balanced mixture of previously correct and incorrect training examples. |

FFN pruning removes intermediate neurons and their associated input/output weights, reducing the FFN intermediate widths. Attention pruning removes whole heads from decoder self-attention and cross-attention, reducing the internal projection widths. Both operations rebuild smaller dense layers. The FFN importance score is inspired by Wanda's use of weights and activations; this implementation prunes neuron groups rather than individual weight elements. The earlier neuron-masking experiments evaluated candidate removals before structural compaction.

The [technical report](docs/technical-report.md) specifies the importance score, removal units, and evaluation conditions. The contribution is the independently designed and implemented competition solution, building on the pretrained models and methods cited in the report.

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
| [docs/technical-report.md](docs/technical-report.md) | Detailed report: problem, design, method, experiments, results, limitations, and references |
| [docs/technical-report.pdf](docs/technical-report.pdf) | Downloadable version of the report |
| [assets/](assets/) | Original overview diagram and result plots in PNG and SVG |
| [results/](results/) | Aggregate CSV tables and notes on units, provenance, and interpretation |

## Materials and Scope

The public release documents the solution and its reported evidence. Implementation notebooks, model weights, raw datasets, and competition presentation files are not included. This release therefore does not provide an executable reproduction package. The figures were newly drawn for this repository; they are not slide screenshots.

The project builds on [InstructBLIP](https://arxiv.org/abs/2305.06500), [A-OKVQA](https://github.com/allenai/aokvqa), [Wanda](https://arxiv.org/abs/2306.11695), [knowledge distillation](https://arxiv.org/abs/1503.02531), and [LoRA](https://arxiv.org/abs/2106.09685). References and the distinction between existing methods and project-specific decisions appear in the report.
