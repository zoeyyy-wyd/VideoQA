# VideoQA: Retrieval-Augmented Video Question Answering

A multimodal video question answering system that retrieves relevant temporal moments before LLM reasoning, producing answers with timestamp-grounded evidence.

## Overview

Given a video and a natural language question, the system:

1. Extracts **CLIP** (semantic) + **SlowFast** (motion) features per 2-second clip
2. Runs **Moment-DETR** to retrieve query-relevant temporal moments
3. Densely samples frames (1 fps) from retrieved moments only
4. **LLM answering pipeline:**

   **Round 1:** A sliding window of 10 frames with a 2-frame overlap runs across the frames. Each window is sent to the LLM, which produces a factual description of up to 500 tokens. All window descriptions for a moment are then reduced into a single per-moment summary by a separate GPT call.

   **Round 2:** All per-moment summaries are combined with the original query and sent to the LLM. The output is the answer plus evidence timestamps.

### Dataset Download
https://www.kaggle.com/datasets/kanzallahhoussam/qvhighlight

### Use the demo

1. Run `server.py` in terminal
2. Open `frontend_page.html` in a browser, set Server URL to `http://localhost:8000`, upload a video, and ask a question

## Configurable Parameters

| Parameter | Default | Description |
|------|------|------|
| `--threshold` | 0.75 | Moment confidence cutoff |
| `--fps_sample` | 1.0 | Frames per second within each moment |
| `--window_size` | 10 | Sliding-window size for LLM calls |
| `--overlap` | 2 | Overlapping frames between adjacent windows |
| `--max_total_frames` | None | Optional global frame cap |

## Evaluation

Moment retrieval on QVHighlights test split (1,542 queries):

| Metric | Score |
|------|------|
| R1@0.5 | 58.30 |
| R1@0.7 | 39.43 |
| mAP@0.5 | 59.01 |
| mAP@0.75 | 35.53 |
| mAP avg (0.5–0.95) | 35.32 |

## Acknowledgments

This project builds on the following prior work:

- **Moment-DETR** and **QVHighlights** dataset:
  Jie Lei, Tamara L. Berg, and Mohit Bansal. *Detecting Moments and Highlights in Videos via Natural Language Queries.* NeurIPS 2021.
  [[paper]](https://arxiv.org/abs/2107.09609) [[code]](https://github.com/jayleicn/moment_detr)

- **CLIP** for semantic visual features:
  Alec Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
  [[paper]](https://arxiv.org/abs/2103.00020)

- **SlowFast** for motion features:
  Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He. *SlowFast Networks for Video Recognition.* ICCV 2019.
  [[paper]](https://arxiv.org/abs/1812.03982)

- **DETR** (architectural foundation of Moment-DETR):
  Nicolas Carion et al. *End-to-End Object Detection with Transformers.* ECCV 2020.
  [[paper]](https://arxiv.org/abs/2005.12872)

## License

For academic use only. Refer to the licenses of the upstream projects (Moment-DETR, CLIP, SlowFast) for their respective terms.
