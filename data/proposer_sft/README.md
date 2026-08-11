# Proposer SFT corpus

This is the complete multimodal corpus used for the Proposer SFT experiment:

- 10,000 examples from 5,000 paired tasks;
- 5,000 formula-proposal examples and 5,000 Critic-conditioned repair examples;
- 10,000 diagnostic PNG images and 5,000 task CSV files;
- no missing referenced assets or exact normalized conversation duplicates.

`proposer_mllm_sft.json.gz` contains portable relative image paths. The image
and CSV assets are stored as `images.tar` and `csv.tar` through Git LFS. Fetch,
unpack and validate everything with:

```bash
git lfs pull
bash run_experiment.sh prepare sft
```

The unpacked JSON, image directory and CSV directory are ignored because they
are generated copies of the tracked archives. The LLaMA-Factory dataset
registration is in `dataset_info.json`; the paper configuration uses a 2%
evaluation split, which produces 9,800 training and 200 evaluation examples.

To reproduce training after installing LLaMA-Factory:

```bash
SFT_BASE_MODEL=Qwen/Qwen3-VL-32B-Instruct \
bash run_experiment.sh train-sft
```
