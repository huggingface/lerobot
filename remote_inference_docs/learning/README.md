# Learning async remote inference

Start with the **[code walkthrough](async_remote_inference_code_walkthrough.html)** to connect the hardware behavior you have observed to the current implementation. Open the HTML in a browser; it works offline and makes no network or robot calls.

The 26 slides are split into five short chapters. Read one chapter at a time, try its questions, then use **Explain & code** when you want the reasoning and exact source excerpts.

| Slides | Chapter | What you should be able to explain afterward |
| --- | --- | --- |
| 1–5 | Map the system | Which process owns motion, model state and each kind of identity. |
| 6–11 | Trace one request | How admission, observation capture, prediction and result acceptance fit together. |
| 12–17 | Read execution | What commitment means; how alignment, blending, refill and RTC differ. |
| 18–22 | Review lifecycle | Why reset, presence loss, terminal faults and planned language holds have different paths. |
| 23–26 | Prepare your review | Which policy contracts, logs and regression tests to inspect first. |

The browser version includes:

- A cursor example: change how many endpoints commit before a chunk arrives and see the usable aligned suffix.
- A timing calculator: change execution length, policy rate, turnaround and refill to explore the playback threshold. This is explanatory arithmetic, not a robot simulator or an automatic tuning rule.
- Embedded source excerpts with line numbers and links into this repository, plus review questions with revealable answers.
- Arrow-key navigation, **N** for explanations, chapter selection and shareable `#slide-N` locations.

The **[PowerPoint companion](async_remote_inference_code_walkthrough.pptx)** has the same diagrams and editable text. Speaker notes contain the explanations, questions, answers and source excerpts. The two interactive examples are static illustrations in PowerPoint.

The **[fully expanded PDF](async_remote_inference_code_walkthrough.pdf)** is a self-contained reading edition, exported on 1 October 2026. It includes every slide, explanation, revealed answer and all 76 source-code tabs, with a linked contents page and chapter/slide bookmarks. The interactive controls are represented by worked static cases. Embedded code remains available without access to this repository.

## Source and scope

The code companion describes commit `c73a3f0403046f669326ca896755b00cbef415fa`, inspected on 30 September 2026. Source excerpts are snapshots, not live includes; consult the named symbols if later edits shift line numbers. Relative source links work when the HTML stays in this directory. Its embedded content works even when copied elsewhere.

This is learning material, not a completed code review or a replacement specification. The [proposal](../async_proposal.md), [progress record](../implementation_progress.md) and [policy support audit](../policy_support_audit.md) retain the decisions, validation evidence and remaining work. Creating this guide did not rerun inference or hardware acceptance tests.

## Earlier high-level overview

The [original browser walkthrough](async_remote_inference_walkthrough.html) and [original PowerPoint](async_remote_inference_walkthrough.pptx) remain useful for the initial UX and motion intuition. They are **historical snapshots from 29 September**, before the aligned request-timing correction was implemented. Their “still planned” timing slide and older lifecycle descriptions must not be read as current status. Use the new code companion for present behavior, including playback-gated alignment, the 10-second default presence grace and configured return after terminal inference faults.
