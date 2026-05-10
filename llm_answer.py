"""
LLM Answer Generation Module

Multi-round approach with sliding-window chunking:
  Round 1: For each moment, slide a window over its frames (with overlap) to get
           per-chunk summaries, then merge them into a per-moment summary.
  Round 2: Combine all per-moment summaries + query → final answer.

Sliding window preserves temporal continuity at chunk boundaries: adjacent
windows share `overlap` frames so actions crossing the boundary are not split.

Usage:
    from llm_answer import LLMAnswerer

    llm = LLMAnswerer(api_key="sk-xxx", model="gpt-4o-mini")
    answer = llm.answer(query, moments, frame_data,
                        max_frames_per_call=10, overlap=2)
"""
import os
from typing import List, Dict, Optional


class LLMAnswerer:
    """Multi-round LLM answer generation with sliding-window chunking."""

    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed")

    def answer(self, query, moments, frame_data=None,
               max_frames_per_call=10, overlap=2):
        """
        Generate answer using multi-round LLM calls with sliding-window chunking.

        Args:
            query: user's question
            moments: list of {"start", "end", "score"}
            frame_data: list of {"moment", "frames", "timestamps"}
            max_frames_per_call: window size (frames per LLM call)
            overlap: number of overlapping frames between adjacent windows

        Returns:
            answer: str
        """
        if not self.client:
            return self._template_answer(query, moments)

        if not frame_data or "4o" not in self.model:
            return self._text_only_answer(query, moments)

        # Round 1: Summarize each moment
        print("  [LLM Round 1] Summarizing each moment...")
        moment_summaries = []
        for i, item in enumerate(frame_data):
            m = item["moment"]
            frames = item["frames"]
            timestamps = item.get("timestamps", [])

            # Sliding-window split
            chunks = self._chunk_frames(frames, timestamps, max_frames_per_call, overlap)

            chunk_summaries = []
            for chunk_frames, chunk_timestamps in chunks:
                summary = self._summarize_moment(
                    query, m, chunk_frames, chunk_timestamps, i + 1
                )
                chunk_summaries.append(summary)

            if len(chunk_summaries) == 1:
                moment_summary = chunk_summaries[0]
            else:
                moment_summary = self._merge_chunk_summaries(
                    query, m, chunk_summaries, i + 1
                )

            moment_summaries.append({"moment": m, "summary": moment_summary})
            print(f"    Moment {i+1} ({m['start']:.1f}s-{m['end']:.1f}s): "
                  f"{len(chunks)} chunk(s) → done")

        # Round 2: Synthesize final answer
        print("  [LLM Round 2] Synthesizing final answer...")
        final_answer = self._synthesize_answer(query, moment_summaries)

        return final_answer

    def _chunk_frames(self, frames, timestamps, max_per_chunk, overlap=2):
        """
        Split frames into sliding windows with overlap.

        window_size = max_per_chunk, stride = max_per_chunk - overlap.
        Last chunk may be shorter than max_per_chunk (truncated to remaining frames).

        Example: 25 frames, window=10, overlap=2 → stride=8
            chunk 0: frames[0:10]   (indices 0-9)
            chunk 1: frames[8:18]   (indices 8-17, shares 2 with chunk 0)
            chunk 2: frames[16:25]  (indices 16-24, shares 2 with chunk 1)
        """
        if len(frames) <= max_per_chunk:
            return [(frames, timestamps)]

        stride = max_per_chunk - overlap
        if stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be < max_per_chunk ({max_per_chunk})")

        chunks = []
        i = 0
        while i < len(frames):
            chunk_f = frames[i:i + max_per_chunk]
            chunk_t = timestamps[i:i + max_per_chunk] if timestamps else []
            chunks.append((chunk_f, chunk_t))

            # Stop once this chunk reaches the end
            if i + max_per_chunk >= len(frames):
                break
            i += stride

        return chunks

    def _summarize_moment(self, query, moment, frames, timestamps, moment_idx):
        """
        Round 1: Summarize a single window's frames.
        """
        content = [{
            "type": "text",
            "text": (
                f"Context: The user asked \"{query}\" about a video.\n"
                f"Below are sequential frames from Segment {moment_idx} "
                f"({moment['start']:.1f}s - {moment['end']:.1f}s).\n"
                f"Describe what you see happening in these frames. "
                f"Focus on details relevant to the user's question. Be factual and specific."
            )
        }]

        for j, b64 in enumerate(frames):
            t = timestamps[j] if j < len(timestamps) else moment["start"]
            content.append({"type": "text", "text": f"[t={t:.1f}s]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _merge_chunk_summaries(self, query, moment, chunk_summaries, moment_idx):
        """
        Merge sliding-window chunk summaries from one moment into a single summary.
        Note: adjacent chunks share `overlap` frames, so some content may be
        described twice — the merge step deduplicates.
        """
        summaries_text = ""
        for i, s in enumerate(chunk_summaries, 1):
            summaries_text += f"Part {i}: {s}\n\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": (
                    f"These are descriptions of overlapping windows over the same video segment "
                    f"({moment['start']:.1f}s - {moment['end']:.1f}s), relevant to the question: \"{query}\"\n\n"
                    f"Adjacent parts share some frames, so deduplicate when content overlaps.\n\n"
                    f"{summaries_text}"
                    f"Combine them into one concise, chronological summary of what happens in this segment."
                )
            }],
            max_tokens=400,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _synthesize_answer(self, query, moment_summaries):
        """Round 2: Combine all moment summaries into a final answer."""
        evidence_text = ""
        for i, ms in enumerate(moment_summaries, 1):
            m = ms["moment"]
            evidence_text += (
                f"Segment {i} ({m['start']:.1f}s - {m['end']:.1f}s, "
                f"relevance: {m['score']:.0%}):\n"
                f"  {ms['summary']}\n\n"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a video analysis assistant. You will receive summaries of "
                        "different video segments that were retrieved as relevant to the user's question.\n\n"
                        "Your task:\n"
                        "1. Give a direct, continuous answer to the question in one paragraph\n"
                        "2. After your answer, on a new line write \"Evidence:\" followed by "
                        "the timestamp ranges that support your answer\n"
                        "3. If the evidence is insufficient, honestly say so"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Segment summaries:\n{evidence_text}\n"
                        f"Please answer the question."
                    )
                }
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _text_only_answer(self, query, moments):
        """Answer without frames (text-only mode, e.g. non-vision models)."""
        evidence = "Retrieved video segments:\n"
        for i, m in enumerate(moments, 1):
            evidence += f"  Segment {i}: {m['start']:.1f}s - {m['end']:.1f}s (relevance: {m['score']:.2%})\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a video analysis assistant. Answer the question based on "
                        "the retrieved segments. Give a direct answer, then list evidence timestamps."
                    )
                },
                {"role": "user", "content": f"Question: {query}\n\n{evidence}\nPlease answer."}
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _template_answer(self, query, moments):
        """Fallback when no API client is available."""
        if not moments:
            return f"No relevant moments found for: '{query}'"
        answer = f"For the query '{query}', the following relevant moments were found:\n\n"
        for i, m in enumerate(moments, 1):
            answer += f"  {i}. {m['start']:.1f}s - {m['end']:.1f}s (relevance: {m['score']:.2%})\n"
        answer += f"\nMost relevant content is at {moments[0]['start']:.1f}s - {moments[0]['end']:.1f}s."
        return answer