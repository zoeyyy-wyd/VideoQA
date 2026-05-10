"""
Video QA Pipeline: Raw Video → Feature Extraction → Moment Retrieval → LLM Answer

End-to-end pipeline:
  Stage 1: Feature Extraction - raw video → CLIP + SlowFast features
  Stage 2: Moment Retrieval   - features + query → moments above confidence threshold
  Stage 3: Frame Extraction   - raw video + moments → dense sequential frames (1 fps, no cap)
  Stage 4: Answer Generation  - query + moments + sequential frames → natural language answer
                                (sliding window with overlap)

Usage:
    # Basic (template answer, no API key needed)
    python3 pipeline.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --query "What is the person doing?" \
        --video /path/to/video.mp4

    # With LLM answer generation
    python3 pipeline.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --query "What is the person doing?" \
        --video /path/to/video.mp4 \
        --openai_key sk-xxx

    # Interactive mode
    python3 pipeline.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --interactive
"""
import os
import sys
import base64
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moment_detr.inference import setup_model
from utils.basic_utils import l2_normalize_np_array
from extract_features import CLIPExtractor, SlowFastExtractor
from llm_answer import LLMAnswerer


# ============================================================
# Stage 1: Feature Extraction
# ============================================================

class FeatureExtractor:
    """Extract CLIP + SlowFast features from a raw video."""

    def __init__(self, device="cuda", use_slowfast=True):
        self.device = device
        self.clip_extractor = CLIPExtractor(device=device)

        self.sf_extractor = None
        if use_slowfast:
            try:
                self.sf_extractor = SlowFastExtractor(device=device)
            except Exception as e:
                print(f"  Warning: SlowFast not available ({e}), using CLIP only")

    def extract(self, video_path, clip_length=2):
        clip_feat, num_clips, duration = self.clip_extractor.extract(video_path, clip_length)

        if self.sf_extractor is not None:
            sf_feat, _, _ = self.sf_extractor.extract(video_path, clip_length)
            min_len = min(len(clip_feat), len(sf_feat))
            video_feat = np.concatenate([sf_feat[:min_len], clip_feat[:min_len]], axis=1)
        else:
            video_feat = clip_feat

        return video_feat, duration

    def extract_text(self, query):
        return self.clip_extractor.extract_text_features(query)


# ============================================================
# Stage 2: Moment Retrieval
# ============================================================

def build_model_inputs(video_feat, text_feat, device="cuda"):
    num_clips = video_feat.shape[0]

    tef_start = np.arange(0, num_clips, 1.0) / num_clips
    tef_end = tef_start + 1.0 / num_clips
    tef = np.stack([tef_start, tef_end], axis=1)
    video_feat_tef = np.concatenate([video_feat, tef], axis=1)

    return {
        "src_vid": torch.from_numpy(video_feat_tef).float().unsqueeze(0).to(device),
        "src_vid_mask": torch.ones(1, num_clips).bool().to(device),
        "src_txt": torch.from_numpy(text_feat).float().unsqueeze(0).to(device),
        "src_txt_mask": torch.ones(1, text_feat.shape[0]).bool().to(device),
    }, num_clips


def merge_overlapping_moments(moments):
    """Merge overlapping or adjacent (within 2s) moments, keeping the higher score."""
    if len(moments) <= 1:
        return moments

    sorted_moments = sorted(moments, key=lambda m: m["start"])

    merged = [sorted_moments[0].copy()]
    for m in sorted_moments[1:]:
        last = merged[-1]
        if m["start"] <= last["end"] + 1.0:
            last["end"] = max(last["end"], m["end"])
            last["score"] = max(last["score"], m["score"])
        else:
            merged.append(m.copy())

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def retrieve_moments(model, video_feat, text_feat, clip_length=2, device="cuda",
                     threshold=0.75, max_moments=10):
    """
    Run Moment-DETR and return moments passing the confidence threshold.

    Pipeline:
      1. Get raw predictions from Moment-DETR
      2. Filter by confidence threshold (fallback to best if none pass)
      3. Merge overlapping moments
      4. Cap at max_moments as a safety limit
    """
    model.eval()
    model_inputs, num_clips = build_model_inputs(video_feat, text_feat, device)

    with torch.no_grad():
        outputs = model(**model_inputs)

    pred_spans = outputs["pred_spans"][0]
    pred_logits = outputs["pred_logits"][0]
    scores = torch.softmax(pred_logits, dim=-1)[:, 0]

    duration = num_clips * clip_length
    moments = []
    for i in range(len(pred_spans)):
        center = pred_spans[i][0].item() * duration
        width = pred_spans[i][1].item() * duration
        moments.append({
            "start": round(max(0, center - width / 2), 1),
            "end": round(min(duration, center + width / 2), 1),
            "score": round(scores[i].item(), 4),
        })

    moments.sort(key=lambda x: x["score"], reverse=True)

    filtered = [m for m in moments if m["score"] >= threshold]

    if not filtered and moments:
        if moments[0]["score"] >= 0.5:
            filtered = [moments[0]]
            print(f"  Warning: no moment above threshold {threshold}, using best match (score: {moments[0]['score']:.4f})")
        else:
            print(f"  No confident match (best score: {moments[0]['score']:.2f})")

    filtered = merge_overlapping_moments(filtered)
    print(f"  After merging overlaps: {len(filtered)} moments")

    filtered = filtered[:max_moments]

    saliency = []
    if "saliency_scores" in outputs:
        saliency = outputs["saliency_scores"][0].cpu().numpy().tolist()

    print(f"  Raw predictions:")
    for m in moments[:10]:
        print(f"    {m['start']:.1f}s-{m['end']:.1f}s ({m['end']-m['start']:.1f}s wide, score {m['score']:.3f})")

    return filtered, saliency


def load_moment_detr(ckpt_path, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = checkpoint["opt"]
    opt.device = torch.device(device)

    model, _, _, _ = setup_model(opt)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


# ============================================================
# Stage 3: Dense Frame Extraction (1 fps, no total cap by default)
# ============================================================

def extract_moment_frames(video_path, moments, fps_sample=1, max_total_frames=None):
    """
    Extract dense frames from retrieved moments to preserve temporal continuity.

    Sampling: np.linspace within each moment, with n_frames = max(3, duration * fps_sample).
    Note: linspace gives uniform distribution across the moment (avg ~1 fps when fps_sample=1),
    not strict integer-second sampling, but guarantees moment boundaries are covered.

    Args:
        video_path: path to video file
        moments: list of {"start", "end", "score"} from Stage 2
        fps_sample: target frames per second within each moment (default 1)
        max_total_frames: optional total cap. None = no limit (default).

    Returns:
        list of {"moment", "frames", "timestamps"}
    """
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frames_used = 0

    for moment in moments:
        if max_total_frames is not None and frames_used >= max_total_frames:
            break

        duration = moment["end"] - moment["start"]
        # Floor at 3 frames so very short moments still have minimal coverage.
        n_frames = max(3, int(duration * fps_sample))

        if max_total_frames is not None:
            n_frames = min(n_frames, max_total_frames - frames_used)

        timestamps = np.linspace(moment["start"], moment["end"], n_frames).tolist()

        frames_b64 = []
        valid_timestamps = []

        for t in timestamps:
            frame_idx = min(int(t * video_fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames_b64.append(base64.b64encode(buf).decode("utf-8"))
                valid_timestamps.append(round(t, 1))

        if frames_b64:
            results.append({
                "moment": moment,
                "frames": frames_b64,
                "timestamps": valid_timestamps,
            })
            frames_used += len(frames_b64)

    cap.release()
    return results


# ============================================================
# Pipeline
# ============================================================

class VideoQAPipeline:
    """End-to-end: raw video + query → answer."""

    def __init__(self, ckpt_path, device="cuda", openai_key=None,
                 llm_model="gpt-4o-mini", use_slowfast=True):
        self.device = device
        self.openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        self.llm_model = llm_model

        print("Initializing pipeline...")
        print(f"  Loading Moment-DETR from {ckpt_path}")
        self.model = load_moment_detr(ckpt_path, device)

        print(f"  Loading feature extractors...")
        self.feat = FeatureExtractor(device=device, use_slowfast=use_slowfast)

        print(f"  Loading LLM answerer ({llm_model})...")
        self.llm = LLMAnswerer(api_key=self.openai_key, model=llm_model)

        print("Pipeline ready!\n")

    def run(self, video_path, query, threshold=0.8,
            fps_sample=1, max_total_frames=None,
            max_frames_per_call=10, overlap=2):
        """
        Run the full pipeline on a raw video.

        Args:
            video_path: path to video file
            query: question about the video
            threshold: confidence threshold (0-1) for moment filtering
            fps_sample: target frames per second within each moment (default 1)
            max_total_frames: optional global frame cap (default None = no limit)
            max_frames_per_call: sliding-window size for LLM calls (default 10)
            overlap: overlapping frames between adjacent windows (default 2)
        """
        # Stage 1
        print(f"[Stage 1] Extracting features from {os.path.basename(video_path)}...")
        video_feat, duration = self.feat.extract(video_path)
        text_feat = self.feat.extract_text(query)
        print(f"  Video: {duration:.1f}s, {video_feat.shape[0]} clips, feature dim {video_feat.shape[1]}")

        # Stage 2
        print(f"[Stage 2] Retrieving moments for: '{query}'")
        moments, saliency = retrieve_moments(
            self.model, video_feat, text_feat, device=self.device, threshold=threshold
        )
        for i, m in enumerate(moments, 1):
            print(f"  {i}. {m['start']:.1f}s - {m['end']:.1f}s (score: {m['score']:.4f})")

        # Stage 3
        cap_str = "no cap" if max_total_frames is None else f"cap={max_total_frames}"
        print(f"[Stage 3] Extracting dense frames (fps_sample={fps_sample}, {cap_str})...")
        frame_data = extract_moment_frames(
            video_path, moments,
            fps_sample=fps_sample,
            max_total_frames=max_total_frames,
        )
        if frame_data:
            n = sum(len(item["frames"]) for item in frame_data)
            print(f"  Extracted {n} sequential frames from {len(frame_data)} moments")

        # Stage 4
        print(f"[Stage 4] Generating answer (sliding window: size={max_frames_per_call}, overlap={overlap})...")
        answer = self.llm.answer(
            query, moments, frame_data,
            max_frames_per_call=max_frames_per_call,
            overlap=overlap,
        )

        return {
            "query": query,
            "answer": answer,
            "moments": moments,
            "saliency": saliency,
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Video QA Pipeline")
    parser.add_argument("--ckpt", required=True, help="Moment-DETR checkpoint")
    parser.add_argument("--video", default=None, help="Video file path")
    parser.add_argument("--query", default=None, help="Question about the video")
    parser.add_argument("--openai_key", default=None, help="OpenAI API key")
    parser.add_argument("--llm", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold (0-1)")
    parser.add_argument("--fps_sample", type=float, default=1.0, help="Frames per second to sample within each moment")
    parser.add_argument("--max_total_frames", type=int, default=None,
                        help="Optional cap on total extracted frames (default: no limit)")
    parser.add_argument("--window_size", type=int, default=10, help="Sliding window size (frames per LLM call)")
    parser.add_argument("--overlap", type=int, default=2, help="Frames overlapping between adjacent windows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_slowfast", action="store_true", help="Skip SlowFast, use CLIP only")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    pipe = VideoQAPipeline(
        ckpt_path=args.ckpt,
        device=args.device,
        openai_key=args.openai_key,
        llm_model=args.llm,
        use_slowfast=not args.no_slowfast,
    )

    run_kwargs = dict(
        threshold=args.threshold,
        fps_sample=args.fps_sample,
        max_total_frames=args.max_total_frames,
        max_frames_per_call=args.window_size,
        overlap=args.overlap,
    )

    if args.interactive:
        print("=" * 50)
        print("  Video QA - Interactive Mode")
        print("  Type 'quit' to exit, 'video' to change video")
        print("=" * 50)

        video_path = None
        while True:
            if video_path is None:
                video_path = input("\nVideo path: ").strip()
                if video_path.lower() == "quit":
                    break
                if not os.path.exists(video_path):
                    print(f"File not found: {video_path}")
                    video_path = None
                    continue

            query = input("Question: ").strip()
            if query.lower() == "quit":
                break
            if query.lower() == "video":
                video_path = None
                continue

            try:
                result = pipe.run(video_path, query, **run_kwargs)
                print(f"\n{'='*50}")
                print(f"Answer:\n{result['answer']}")
                print(f"{'='*50}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        if not args.video or not args.query:
            parser.print_help()
            print("\nError: --video and --query are required")
            return

        result = pipe.run(args.video, args.query, **run_kwargs)

        print(f"\n{'='*50}")
        print(f"Question: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nRetrieved Moments:")
        for i, m in enumerate(result["moments"], 1):
            print(f"  {i}. {m['start']:.1f}s - {m['end']:.1f}s (score: {m['score']:.4f})")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()