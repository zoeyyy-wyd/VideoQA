"""
Video QA API Server

FastAPI backend that connects the React frontend to the Moment-DETR pipeline.

Usage:
    pip install fastapi uvicorn python-multipart
    python3 server.py --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt
"""
import os
import sys
import shutil
import tempfile
import argparse
import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import VideoQAPipeline
from llm_answer import LLMAnswerer

# ============================================================
# App
# ============================================================

app = FastAPI(title="Video QA API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None      # initialized in main()
defaults = {}    # server-wide default run params, also set in main()


@app.post("/query")
async def query_video(
    video: UploadFile = File(...),
    query: str = Form(...),
    # All params below are optional per-request overrides.
    # If not provided, the server-wide defaults (set via CLI) are used.
    threshold: Optional[float] = Form(None),
    fps_sample: Optional[float] = Form(None),
    window_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    max_total_frames: Optional[int] = Form(None),
    openai_key: Optional[str] = Form(None),
):
    """
    Upload a video and ask a question.
    Returns retrieved moments and an LLM-generated answer.
    """
    # Resolve params: per-request override → server default
    run_kwargs = dict(
        threshold=threshold if threshold is not None else defaults["threshold"],
        fps_sample=fps_sample if fps_sample is not None else defaults["fps_sample"],
        max_frames_per_call=window_size if window_size is not None else defaults["window_size"],
        overlap=overlap if overlap is not None else defaults["overlap"],
        max_total_frames=max_total_frames if max_total_frames is not None else defaults["max_total_frames"],
    )

    # Save uploaded video to temp file
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    # Per-request LLM key override (single-threaded only — see warning in startup log)
    original_llm = pipe.llm
    if openai_key:
        pipe.llm = LLMAnswerer(api_key=openai_key, model=pipe.llm_model)

    try:
        result = pipe.run(tmp_path, query, **run_kwargs)
        return JSONResponse({
            "query": result["query"],
            "answer": result["answer"],
            "moments": result["moments"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipe.llm = original_llm
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "defaults": defaults,
    }


# ============================================================
# Main
# ============================================================

def main():
    global pipe, defaults

    parser = argparse.ArgumentParser(description="Video QA API Server")

    # Model / hardware
    parser.add_argument("--ckpt", required=True, help="Moment-DETR checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_slowfast", action="store_true")

    # LLM
    parser.add_argument("--openai_key", default=None,
                        help="OpenAI API key (server-wide default; can be overridden per-request)")
    parser.add_argument("--llm", default="gpt-4o-mini", help="LLM model name")

    # Server-wide defaults for run() params
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Default moment confidence threshold (0-1)")
    parser.add_argument("--fps_sample", type=float, default=2.0,
                        help="Default frames per second within each moment")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Default sliding-window size for LLM calls")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Default overlap (frames) between adjacent windows")
    parser.add_argument("--max_total_frames", type=int, default=None,
                        help="Default global cap on extracted frames (default: no limit)")

    # Network
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    # Stash defaults for /query to consult
    defaults = {
        "threshold": args.threshold,
        "fps_sample": args.fps_sample,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "max_total_frames": args.max_total_frames,
    }

    print("Initializing pipeline...")
    pipe = VideoQAPipeline(
        ckpt_path=args.ckpt,
        device=args.device,
        openai_key=args.openai_key,
        llm_model=args.llm,
        use_slowfast=not args.no_slowfast,
    )

    print("\nServer-wide defaults:")
    for k, v in defaults.items():
        print(f"  {k}: {v}")

    print(f"\nServer starting at http://localhost:{args.port}")
    print(f"API docs at http://localhost:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()