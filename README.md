# LLM4HLS Agent Project

This repository contains experiments for generating, debugging and
optimising Xilinx HLS code using large language models.

## Current objectives

1. Establish a reproducible Vitis HLS workflow.
2. Test prompt-based repair of deliberately faulty HLS implementations.
3. Reproduce an existing LLM-based HLS generation workflow.
4. Extend the workflow using structured compiler, simulation and
   synthesis feedback.

## Repository structure

- `benchmarks/`: HLS source code and testbenches
- `experiments/`: individual experimental runs and metadata
- `prompts/`: exact prompts supplied to language models
- `results/`: selected simulation and synthesis results
- `scripts/`: experiment automation and report parsing
- `notes/`: environment details, decisions and laboratory logs

## First experiment

The first experiment uses a simple integer adder to validate the complete
generate-test-feedback-repair workflow.