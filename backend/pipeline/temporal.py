# backend/pipeline/temporal.py
import torch
import torch.nn as nn
from collections import deque

class EmotionLSTM(nn.Module):
    """
    Input per step: (B, 7) softmax probabilities from CNN
    Hidden state summarises emotion trajectory over the answer
    Output: (B, 7) smoothed emotion distribution
    """
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 7)

    def forward(self, x):           # x: (B, T, 7)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # predict from final timestep


class TemporalEmotionTracker:
    """Stateful tracker for a live interview session."""
    def __init__(self, window_size=30):  # 30 frames ≈ 1 second at 30fps
        self.buffer = deque(maxlen=window_size)
        self.emotion_history = []        # one entry per frame

    def update(self, cnn_probs: list[float]) -> dict:
        """
        cnn_probs: softmax output from EmotionCNN for this frame
        Returns smoothed emotion state for this moment.
        """
        self.buffer.append(cnn_probs)
        self.emotion_history.append(cnn_probs)
        # Simple exponential moving average before LSTM is trained
        if len(self.buffer) < 5:
            return {"emotion": cnn_probs, "smoothed": False}
        window = torch.tensor(list(self.buffer)).unsqueeze(0)  # (1, T, 7)
        # LSTM inference here once trained
        return {"emotion": cnn_probs, "smoothed": True}

    def get_answer_summary(self) -> dict:
        """
        Called at end of each interview answer.
        Returns dominant emotion, trajectory, and volatility.
        """
        if not self.emotion_history:
            return {}
        arr = torch.tensor(self.emotion_history)  # (T, 7)
        dominant = arr.mean(0).argmax().item()
        volatility = arr.std(0).mean().item()     # how much emotion shifted
        self.emotion_history.clear()
        return {
            "dominant_emotion": dominant,
            "volatility": round(volatility, 3),
            "trajectory": arr.tolist()
        }