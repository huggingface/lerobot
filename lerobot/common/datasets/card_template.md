---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

{{ dataset_description | default("", true) }}

- **Homepage:** {{ url | default("[More Information Needed]", true)}}
- **Paper:** {{ paper | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}

## Dataset Structure

{{ dataset_structure | default("[More Information Needed]", true)}}

## Citation

**BibTeX:**

```bibtex
{{ citation_bibtex | default("[More Information Needed]", true)}}
```
