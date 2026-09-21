#!/usr/bin/env python3
"""
PyAnnote speaker diarization script.
Processes audio files to identify and separate different speakers.
"""

import argparse
import json
import sys
import os
import time
import traceback
from pathlib import Path
from pyannote.audio import Pipeline
import torch
import torchaudio

# Fix for PyTorch 2.6+ which defaults weights_only=True
# We need to allowlist PyAnnote's custom classes
try:
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")


def load_audio_for_pipeline(audio_path: str):
    """
    Load audio before calling PyAnnote so pyannote.audio 4.x does not rely on
    torchcodec's internal AudioDecoder. The input is already converted to WAV
    by Scriberr, and torchaudio handles that reliably in this environment.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "uri": Path(audio_path).stem,
    }


def build_silero_vad_audio(audio_input, onset: float = None, offset: float = None):
    """
    Build a compact speech-only waveform using the locally packaged Silero model.
    Returns the compacted input and a timeline used to restore original timestamps.
    """
    from silero_vad import get_speech_timestamps, load_silero_vad

    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]
    if waveform.numel() == 0:
        return audio_input, []

    mono = waveform.mean(dim=0) if waveform.ndim == 2 else waveform.reshape(-1)
    vad_sample_rate = 16000
    vad_waveform = mono
    if sample_rate != vad_sample_rate:
        vad_waveform = torchaudio.functional.resample(mono, sample_rate, vad_sample_rate)

    model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(
        vad_waveform,
        model,
        sampling_rate=vad_sample_rate,
        threshold=onset if onset is not None else 0.5,
        neg_threshold=offset if offset is not None else 0.35,
        min_speech_duration_ms=200,
        min_silence_duration_ms=300,
        speech_pad_ms=150,
    )
    if not speech_timestamps:
        return audio_input, []

    chunks = []
    timeline = []
    compact_cursor = 0
    for speech_range in speech_timestamps:
        start = round(speech_range["start"] * sample_rate / vad_sample_rate)
        end = round(speech_range["end"] * sample_rate / vad_sample_rate)
        start = max(0, start)
        end = min(mono.shape[0], end)
        if end <= start:
            continue
        chunk = waveform[:, start:end] if waveform.ndim == 2 else waveform[start:end].unsqueeze(0)
        chunks.append(chunk)
        duration = (end - start) / sample_rate
        timeline.append({
            "compact_start": compact_cursor / sample_rate,
            "compact_end": compact_cursor / sample_rate + duration,
            "original_start": start / sample_rate,
            "original_end": end / sample_rate,
        })
        compact_cursor += end - start

    if not chunks:
        return audio_input, []

    compact_waveform = torch.cat(chunks, dim=1)
    return {
        "waveform": compact_waveform,
        "sample_rate": sample_rate,
        "uri": audio_input["uri"],
    }, timeline


def restore_original_time(value: float, timeline):
    if not timeline:
        return value
    for item in timeline:
        if item["compact_start"] <= value <= item["compact_end"]:
            return item["original_start"] + (value - item["compact_start"])
    if value < timeline[0]["compact_start"]:
        return timeline[0]["original_start"]
    return timeline[-1]["original_end"]


def remap_segments_to_original_time(segments, timeline):
    if not timeline:
        return segments
    remapped = []
    for segment in segments:
        for region in timeline:
            compact_start = max(segment["start"], region["compact_start"])
            compact_end = min(segment["end"], region["compact_end"])
            if compact_end <= compact_start:
                continue
            item = dict(segment)
            item["start"] = region["original_start"] + compact_start - region["compact_start"]
            item["end"] = region["original_start"] + compact_end - region["compact_start"]
            item["duration"] = item["end"] - item["start"]
            remapped.append(item)
    return remapped


def diarize_audio(
    audio_path: str,
    output_file: str,
    hf_token: str,
    model: str = "pyannote/speaker-diarization-community-1",
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    output_format: str = "rttm",
    device: str = "auto",
    segmentation_onset: float = None,
    segmentation_offset: float = None,
    pre_vad_method: str = "pyannote",
):
    """
    Perform speaker diarization on audio file using PyAnnote.
    """
    print(f"Loading PyAnnote speaker diarization pipeline: {model}")

    try:
        # Initialize the diarization pipeline
        pipeline = Pipeline.from_pretrained(
            model,
            token=hf_token
        )

        # Move to the requested device. Never silently fall back when CUDA was
        # explicitly requested: that can turn a short GPU job into a very long
        # CPU job without making the cause visible.
        try:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for diarization but is not available to PyTorch")
            if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
                pipeline = pipeline.to(torch.device("cuda"))
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                print(f"Using CUDA for diarization: {gpu_name}")
            else:
                pipeline = pipeline.to(torch.device("cpu"))
                print("Using CPU for diarization")
        except Exception as e:
            print(f"Error selecting diarization device: {e}")
            raise

        # Apply segmentation thresholds if provided
        if segmentation_onset is not None or segmentation_offset is not None:
            try:
                # Get current parameters
                params = pipeline.parameters(instantiated=True)

                # Community-1 does not necessarily expose the legacy
                # segmentation hyperparameters. Only update parameters that
                # the loaded pipeline actually declares.
                if "segmentation" in params:
                    segmentation_params = params["segmentation"]
                    changed = False
                    if segmentation_onset is not None and "threshold" in segmentation_params:
                        params["segmentation"]["threshold"] = segmentation_onset
                        print(f"Set segmentation onset threshold: {segmentation_onset}")
                        changed = True
                    elif segmentation_onset is not None:
                        print("Segmentation onset is not supported by this pipeline; using its trained default")
                    if segmentation_offset is not None and "min_duration_off" in segmentation_params:
                        params["segmentation"]["min_duration_off"] = segmentation_offset
                        print(f"Set segmentation offset (min_duration_off): {segmentation_offset}")
                        changed = True
                    elif segmentation_offset is not None:
                        print("Segmentation offset is not supported by this pipeline; using its trained default")

                    if changed:
                        pipeline.instantiate(params)
                else:
                    print("Segmentation tuning is not supported by this pipeline; using its trained defaults")
            except Exception as e:
                print(f"Warning: Could not set segmentation thresholds: {e}")
                print("Continuing with default thresholds")

        print("Pipeline loaded successfully")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Make sure you have a valid Hugging Face token and have accepted the model's license")
        sys.exit(1)

    print(f"Processing audio file: {audio_path}")

    try:
        audio_input = load_audio_for_pipeline(audio_path)
        original_duration = audio_input["waveform"].shape[1] / audio_input["sample_rate"]
        vad_timeline = []
        if pre_vad_method == "silero":
            audio_input, vad_timeline = build_silero_vad_audio(
                audio_input,
                onset=segmentation_onset,
                offset=segmentation_offset,
            )
            compact_duration = audio_input["waveform"].shape[1] / audio_input["sample_rate"]
            print("Pre-diarization VAD: method=silero")
            print(f"  Speech regions: {len(vad_timeline)}")
            print(f"  Original duration: {original_duration:.2f} seconds")
            print(f"  Diarization duration: {compact_duration:.2f} seconds")
        elif pre_vad_method == "none":
            print("Pre-diarization VAD disabled")
        else:
            print("Pre-diarization VAD: method=pyannote_internal")

        # Run diarization
        diarization_params = {}
        if num_speakers is not None:
            diarization_params["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers

        inference_started = time.monotonic()
        print(f"Starting diarization inference ({original_duration:.2f} seconds of source audio)")
        if diarization_params:
            print(f"Using speaker constraints: {diarization_params}")
            diarization = pipeline(audio_input, **diarization_params)
        else:
            print("Using automatic speaker detection")
            diarization = pipeline(audio_input)

        print(f"Diarization inference completed in {time.monotonic() - inference_started:.1f} seconds")
        print(f"Saving results to: {output_file}")

        if output_format == "rttm":
            # Save the diarization output to RTTM format
            annotation = getattr(diarization, "speaker_diarization", diarization)
            with open(output_file, "w") as rttm:
                annotation.write_rttm(rttm)
        else:
            # Save as JSON format
            save_json_format(diarization, output_file, audio_path, vad_timeline, pre_vad_method)

        # Print summary
        speakers = set()
        total_speech_time = 0.0

        # Iterate over speaker diarization
        # PyAnnote 4.x returns a DiarizeOutput object with a speaker_diarization attribute
        if hasattr(diarization, "speaker_diarization"):
            for turn, speaker in diarization.speaker_diarization:
                speakers.add(speaker)
                total_speech_time += turn.duration
        elif hasattr(diarization, "itertracks"):
            # Fallback for older versions
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                total_speech_time += segment.duration
        else:
            # Try iterating directly (some versions return Annotation directly)
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                total_speech_time += segment.duration

        print(f"\nDiarization Summary:")
        print(f"  Speakers detected: {len(speakers)}")
        print(f"  Speaker labels: {sorted(speakers)}")
        print(f"  Total speech time: {total_speech_time:.2f} seconds")
        print(f"  Output file saved: {output_file}")

    except Exception as e:
        print(f"Error during diarization: {e}")
        traceback.print_exc()
        sys.exit(1)


def save_json_format(diarization, output_file: str, audio_path: str, vad_timeline=None, pre_vad_method: str = "pyannote"):
    """Save diarization results in JSON format."""
    vad_timeline = vad_timeline or []
    segments = annotation_to_segments(getattr(diarization, "speaker_diarization", diarization))
    exclusive_segments = annotation_to_segments(getattr(diarization, "exclusive_speaker_diarization", None))
    segments = remap_segments_to_original_time(segments, vad_timeline)
    exclusive_segments = remap_segments_to_original_time(exclusive_segments, vad_timeline)
    if not exclusive_segments:
        exclusive_segments = segments

    speakers = {segment["speaker"] for segment in segments}
    speakers.update(segment["speaker"] for segment in exclusive_segments)

    # Sort segments by start time
    segments.sort(key=lambda x: x["start"])
    exclusive_segments.sort(key=lambda x: x["start"])

    results = {
        "audio_file": audio_path,
        "model": "pyannote/speaker-diarization-community-1",
        "segments": segments,
        "exclusive_segments": exclusive_segments,
        "speakers": sorted(speakers),
        "speaker_count": len(speakers),
        "total_duration": max(seg["end"] for seg in segments) if segments else 0,
        "processing_info": {
            "total_segments": len(segments),
            "exclusive_segments": len(exclusive_segments),
            "total_speech_time": sum(seg["duration"] for seg in segments),
            "pre_diarization_vad": pre_vad_method,
            "pre_diarization_vad_regions": len(vad_timeline)
        }
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def annotation_to_segments(annotation):
    """Convert a pyannote Annotation-like object into serializable segments."""
    if annotation is None:
        return []

    segments = []
    speakers = set()

    if hasattr(annotation, "itertracks"):
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "confidence": 1.0,
                "duration": segment.duration
            })
            speakers.add(speaker)
    else:
        try:
            for turn, speaker in annotation:
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "confidence": 1.0,
                    "duration": turn.duration
                })
                speakers.add(speaker)
        except TypeError:
            return []

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Perform speaker diarization using PyAnnote.audio"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--hf-token",
        required=True,
        help="Hugging Face access token"
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
        help="PyAnnote model to use"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers"
    )
    parser.add_argument(
        "--output-format",
        choices=["rttm", "json"],
        default="rttm",
        help="Output format"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--segmentation-onset",
        type=float,
        help="Voice activity detection onset threshold (0.0-1.0). Lower values detect quieter speech."
    )
    parser.add_argument(
        "--segmentation-offset",
        type=float,
        help="Voice activity detection offset/min_duration_off (0.0-1.0). Lower values are more sensitive to speech endings."
    )
    parser.add_argument(
        "--pre-vad-method",
        choices=["pyannote", "silero", "none"],
        default="pyannote",
        help="Local pre-diarization VAD. Silero uses the model bundled in the installed Python package."
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Validate speaker constraints
    if args.min_speakers is not None and args.min_speakers < 1:
        print("Error: min_speakers must be at least 1")
        sys.exit(1)

    if args.max_speakers is not None and args.max_speakers < 1:
        print("Error: max_speakers must be at least 1")
        sys.exit(1)

    if args.num_speakers is not None and args.num_speakers < 1:
        print("Error: num_speakers must be at least 1")
        sys.exit(1)

    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        print("Error: num_speakers cannot be combined with min_speakers or max_speakers")
        sys.exit(1)

    if (args.min_speakers is not None and args.max_speakers is not None and
        args.min_speakers > args.max_speakers):
        print("Error: min_speakers cannot be greater than max_speakers")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        diarize_audio(
            audio_path=args.audio_file,
            output_file=args.output,
            hf_token=args.hf_token,
            model=args.model,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            output_format=args.output_format,
            device=args.device,
            segmentation_onset=args.segmentation_onset,
            segmentation_offset=args.segmentation_offset,
            pre_vad_method=args.pre_vad_method,
        )
    except Exception as e:
        print(f"Error during diarization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
