import base64
import json
import os
import uuid

try:
    from djl_python import Input, Output
except ImportError:
    Input = None
    Output = None


def _load_whisper_pipeline():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    model_id = "openai/whisper-large-v3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    from transformers import pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=5,
        device=device,
    )
    return pipe


def _load_diarization_pipeline():
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable required for pyannote.audio")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if __import__("torch").cuda.is_available():
        pipeline = pipeline.to("cuda")
    return pipeline


_whisper_pipe = None
_diarization_pipe = None


def load_model():
    global _whisper_pipe, _diarization_pipe
    _whisper_pipe = _load_whisper_pipeline()
    _diarization_pipe = _load_diarization_pipeline()


def _preprocess(input_data: dict) -> str:
    audio_b64 = input_data.get("audio_b64")
    if not audio_b64:
        raise ValueError("Missing 'audio_b64' in request")
    audio_bytes = base64.b64decode(audio_b64)
    file_id = str(uuid.uuid4())
    filepath = f"/tmp/audio_{file_id}.wav"
    try:
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        return filepath
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise e


def _align_segments_to_speakers(
    whisper_segments: list[dict],
    diarization_segments: list[tuple],
) -> list[dict]:
    result = []
    for seg in whisper_segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = "SPEAKER_00"
        best_overlap = 0.0
        for (d_start, d_end, d_speaker) in diarization_segments:
            overlap_start = max(start, d_start)
            overlap_end = min(end, d_end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    speaker = d_speaker
        result.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return result


def _inference(filepath: str) -> list[dict]:
    global _whisper_pipe, _diarization_pipe
    if _whisper_pipe is None or _diarization_pipe is None:
        load_model()

    try:
        out = _whisper_pipe(filepath, return_timestamps="word", generate_kwargs={"language": None})
        whisper_segments = []
        if isinstance(out, dict) and "chunks" in out:
            for chunk in out["chunks"]:
                whisper_segments.append({
                    "start": chunk.get("timestamp", (0, 0))[0] or 0.0,
                    "end": chunk.get("timestamp", (0, 0))[1] or 0.0,
                    "text": chunk.get("text", ""),
                })
        elif isinstance(out, dict) and "text" in out:
            text = out["text"]
            if text:
                whisper_segments.append({"start": 0.0, "end": 0.0, "text": text})
        else:
            whisper_segments.append({"start": 0.0, "end": 0.0, "text": str(out)})

        diarization = _diarization_pipe(filepath)
        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append((turn.start, turn.end, speaker))

        aligned = _align_segments_to_speakers(whisper_segments, diarization_segments)
        return aligned
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


def _postprocess(segments: list[dict]) -> str:
    return json.dumps(segments)


def handle(inputs):
    filepath = None
    try:
        if Input and hasattr(inputs, "get_as_json"):
            data = inputs.get_as_json()
        elif isinstance(inputs, dict):
            data = inputs
        else:
            data = json.loads(inputs) if isinstance(inputs, (str, bytes)) else {}
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        filepath = _preprocess(data)
        segments = _inference(filepath)
        return _postprocess(segments)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
