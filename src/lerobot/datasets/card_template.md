---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
# prettier-ignore
{{card_data}}
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

{% if repo_id is defined and repo_id %}
<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path={{ repo_id }}">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>
{% endif %}

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
