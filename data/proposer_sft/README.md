# Proposer SFT Data

This directory contains the complete synthetic 10,000-example Proposer SFT data
package used for VL-LoopSR.

## Main Split

`raw_10k/` contains:

- `train.json`
  - 10,000 multimodal SFT records.
  - Each record has `conversations`, `images`, `data_csv`, `dimension`, and
    `target_expression` fields.
  - Image and CSV paths are relative to `raw_10k/`.

- `manifest.csv`
  - One row per example with id, split, dimension, target expression, CSV path,
    and image path.

- `summary.csv`
  - Split-level counts.

- `images/`
  - 10,000 generated visualization PNG files.

- `csv/`
  - 10,000 sampled numeric data tables.

## Counts

| Split | Records | Images | CSV files |
|---|---:|---:|---:|
| `1d` | 5,000 | 5,000 | 5,000 |
| `2d` | 3,000 | 3,000 | 3,000 |
| `hd` | 2,000 | 2,000 | 2,000 |
| **Total** | **10,000** | **10,000** | **10,000** |

## Format

Example record:

```json
{
  "id": "SmartGen-1_run_001",
  "split": "1d",
  "dimension": 1,
  "target_expression": "-1.2*sin(x1)",
  "data_csv": "csv/1d/SmartGen-1/SmartGen-1_run_001.csv",
  "conversations": [
    {"from": "human", "value": "<image>..."},
    {"from": "gpt", "value": "Final formula: $$y = -1.2*sin(x1)$$"}
  ],
  "images": ["images/1d/SmartGen-1/SmartGen-1_run_001.png"]
}
```

## Notes

- The data is synthetic and contains no API keys, local absolute paths, or model
  weights.
- PNG images are marked for Git LFS in the repository `.gitattributes`.
- The older distilled teacher-response subsets are not the main public SFT
  package; this directory now points to the complete 10k data.
