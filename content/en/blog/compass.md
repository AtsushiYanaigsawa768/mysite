---
title: "Developing a Japanese Financial VLM through Integrated Reasoning and Document Comprehension"
date: 2026-03-09
summary: "We explore the development of a Vision-Language Model tailored for the Japanese financial domain, combining reasoning enhancement techniques with advanced document reading capabilities."
tags: ["VLM", "Finance", "NLP", "Japanese"]
layout: "blog-single"
tocSections:
  - "Background"
  - "Approach"
  - "Experiments"
  - "Results"
  - "Conclusion"
ShowToc: true
---

## Background

The Japanese financial industry generates vast amounts of documents—annual reports, earnings summaries, regulatory filings—that require both visual understanding and domain-specific reasoning. While recent Vision-Language Models (VLMs) have shown impressive capabilities in general tasks, their performance on Japanese financial documents remains limited due to the unique challenges of this domain: complex table layouts, mixed Japanese-English text, and the need for numerical reasoning.

In this work, we set out to build a specialized VLM that can not only *read* Japanese financial documents but also *reason* about their contents in a financially meaningful way.

## Approach

Our approach integrates two key components:

### Reasoning Enhancement

We employ a multi-stage training pipeline that progressively strengthens the model's reasoning capabilities:

1. **Pre-training on financial corpora**: We curate a large-scale dataset of Japanese financial texts, including annual securities reports (有価証券報告書), earnings presentations, and analyst reports.
2. **Chain-of-thought fine-tuning**: We generate step-by-step reasoning traces for financial QA tasks using a stronger teacher model, then distill these into our target VLM.
3. **Reinforcement learning from financial feedback**: Domain experts provide preference signals on model outputs, which we use for RLHF-style optimization.

### Document Comprehension

For robust document understanding, we focus on:

- **Layout-aware encoding**: We integrate spatial position embeddings that capture the 2D structure of financial documents, enabling the model to correctly associate headers with their corresponding data cells in tables.
- **OCR-free reading**: Our architecture directly processes document images without relying on external OCR pipelines, reducing error propagation.
- **Multi-page context**: Financial documents often span dozens of pages. We implement an efficient attention mechanism that maintains context across page boundaries.

## Experiments

We evaluate our model on several benchmarks:

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| JFinQA | Japanese financial question answering | Accuracy |
| DocTabQA | Table understanding in financial docs | F1 |
| FinRead-JP | Reading comprehension on earnings reports | ROUGE-L |
| NumReason | Numerical reasoning over financial data | Exact Match |

### Training Details

- **Base model**: A 7B parameter VLM pre-trained on a mixture of Japanese and English web data
- **Financial pre-training**: 50B tokens of Japanese financial text
- **Fine-tuning**: 100K instruction-response pairs with chain-of-thought annotations
- **Hardware**: 8× A100 GPUs, training completed in approximately 2 weeks

## Results

Our model, which we call **Compass**, achieves state-of-the-art results across all four benchmarks:

- **JFinQA**: 78.3% accuracy (+12.1% over the previous best)
- **DocTabQA**: 85.7 F1 (+8.4 over baseline)
- **FinRead-JP**: 71.2 ROUGE-L (+6.8 over baseline)
- **NumReason**: 64.5% exact match (+15.2% over baseline)

The largest improvements come from numerical reasoning tasks, where our chain-of-thought training proves especially effective. The model demonstrates the ability to perform multi-step calculations involving currency conversion, percentage changes, and ratio analysis—capabilities that are critical for real-world financial analysis.

### Ablation Study

We conduct ablation studies to understand the contribution of each component:

- Removing reasoning enhancement reduces JFinQA accuracy by 9.2%
- Removing layout-aware encoding reduces DocTabQA F1 by 11.3
- Using external OCR instead of OCR-free reading introduces a 4.1% error rate increase

## Conclusion

We have presented Compass, a Japanese financial VLM that integrates reasoning enhancement with document comprehension capabilities. Our results demonstrate that domain-specific training combined with architecture innovations can significantly improve performance on challenging financial document understanding tasks.

Future directions include extending the model to handle real-time market data, supporting longer document contexts, and exploring applications in automated financial reporting and compliance checking.
