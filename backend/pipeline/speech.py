# backend/pipeline/speech.py
import whisper
import librosa
import numpy as np

whisper_model = whisper.load_model("base")   # ~150MB, runs locally

def transcribe_audio_chunk(audio_bytes: bytes, sample_rate=16000) -> str:
    """Transcribe a chunk of audio (one interview answer)."""
    audio_arr = np.frombuffer(audio_bytes, dtype=np.float32)
    result = whisper_model.transcribe(audio_arr, language="en")
    return result["text"].strip()

def extract_acoustic_features(audio_bytes: bytes, sample_rate=16000) -> dict:
    """
    9417 Wk 2-3: extract a feature vector from audio for classical classification.
    Returns a dict of interpretable acoustic features.
    """
    y = np.frombuffer(audio_bytes, dtype=np.float32)

    # Pitch (F0) via librosa
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sample_rate)
    f0 = f0[~np.isnan(f0)]

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)

    # Energy
    rms = librosa.feature.rms(y=y)[0]

    return {
        "pitch_mean":   float(f0.mean()) if len(f0) > 0 else 0.0,
        "pitch_std":    float(f0.std())  if len(f0) > 0 else 0.0,
        "energy_mean":  float(rms.mean()),
        "energy_std":   float(rms.std()),
        "mfcc_mean":    mfccs.mean(axis=1).tolist(),  # 13 values
    }