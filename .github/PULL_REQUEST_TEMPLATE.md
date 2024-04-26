# What does this PR do?

Examples:
- Fixes # (issue)
- Adds new dataset
- Optimizes something

## How was it tested?

Examples:
- Added `test_something` in `tests/test_stuff.py`.
- Added `new_feature` and checked that training converges with policy X on dataset/environment Y.
- Optimized `some_function`, it now runs X times faster than previously.

## How to checkout & try? (for the reviewer)

Examples:
```bash
DATA_DIR=tests/data pytest -sx tests/test_stuff.py::test_something
```
```bash
python lerobot/scripts/train.py --some.option=true
```

## Before submitting
Please read the [contributor guideline](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md#submitting-a-pull-request-pr).


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR. Try to avoid tagging more than 3 people.
