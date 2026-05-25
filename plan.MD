# Interview Coach — Build Plan

**Project:** A SaaS web app that conducts structured interviews via the browser, analyses
facial emotion (video), transcribes speech (audio), evaluates answer quality (language),
fuses all three signals, and produces a post-interview report for recruiters.

**Learning goal:** Cover ~90% of COMP9417, COMP9444, and COMP9517 through a single
coherent build — understanding the theory *as* you implement it, not separately from it.

---

## Stack decisions (locked before building)

| Layer | Technology | Why |
|---|---|---|
| Backend | Python 3.10 + FastAPI | Async WebSocket support, easy to deploy |
| ML runtime | PyTorch | Required by 9444; best ecosystem |
| Face pipeline | OpenCV + MediaPipe | OpenCV for 9517 concepts; MediaPipe for production face mesh |
| Transcription | OpenAI Whisper (local, `whisper` library) | Free, runs locally, no API key needed |
| LLM scoring | Anthropic Claude API | Answer quality + coaching tip generation |
| Frontend | Vanilla HTML/CSS/JS | No framework overhead; WebRTC built into browser |
| Database | SQLite (via SQLAlchemy) | Zero config, sufficient for demo scale |
| Report UI | Jinja2 templates + Chart.js | Server-rendered, no build step |

---

## Phase overview

```
Phase 0 — Environment & architecture         (Day 1)
Phase 1 — Classical CV pipeline              (Days 2–3)   9517 Wk 1–5
Phase 2 — Classical ML classifiers           (Days 3–4)   9417 Wk 1–6
Phase 3 — Deep emotion CNN                   (Days 5–6)   9444 Wk 1–3, 9517 Wk 7–8
Phase 4 — Temporal modelling                 (Day 7)      9444 Wk 5, 9517 Wk 9
Phase 5 — Speech + language stream           (Days 8–9)   9444 Wk 5, 9417 Wk 2–3
Phase 6 — Fusion layer                       (Day 10)     9417 Wk 6
Phase 7 — Browser frontend + WebSocket       (Days 11–12) 9517 Wk 1 (image formation)
Phase 8 — Recruiter dashboard & report       (Days 13–14) 9417 Wk 9 (clustering)
```
---

## Topic coverage summary

| Topic | Course | Phase | Status |
|---|---|---|---|
| Image formation | 9517 Wk 1 | Phase 1 | Core |
| Image filtering & enhancement | 9517 Wk 2 | Phase 1 | Core |
| Feature representation (HOG) | 9517 Wk 3 | Phase 1 | Core |
| Pattern recognition | 9517 Wk 4 | Phase 1 | Core |
| Image segmentation | 9517 Wk 5 | Phase 1 | Core |
| Deep learning for CV | 9517 Wk 7–8 | Phase 3 | Core |
| Motion tracking | 9517 Wk 9 | Phase 4 | Core |
| Regression & classification | 9417 Wk 1–3 | Phase 2 | Core |
| Decision trees | 9417 Wk 4 | Phase 2 | Core |
| SVMs & kernel methods | 9417 Wk 5 | Phase 2 | Core |
| Ensemble methods | 9417 Wk 6 | Phase 6 | Core |
| Unsupervised / k-means | 9417 Wk 9 | Phase 6 | Core |
| Learning theory | 9417 Wk 10 | Phase 2–3 | Embedded |
| Perceptrons & backprop | 9444 Wk 1–2 | Phase 3 | Core |
| Deep CNNs | 9444 Wk 3 | Phase 3 | Core |
| Geometric analysis | 9444 Wk 4 | Phase 3 | Add-on (Grad-CAM) |
| RNNs, LSTMs, language | 9444 Wk 5 | Phase 4–5 | Core |
| Deep RL | 9444 Wk 6 | — | Study separately |
| Autoencoders & GANs | 9444 Wk 7 | — | Stretch goal |
| Designing DL applications | 9444 Wk 8 | All phases | Meta-goal |

**Coverage: ~90% of syllabus across all three courses**

---

## Stretch goals (if time permits)

- **Autoencoder anomaly detection** — train an autoencoder on "neutral" face
  crops; high reconstruction error during an answer signals unusual affect
- **GAN data augmentation** — synthesise additional training examples for
  underrepresented emotion classes (disgust, fear) in FER-2013
- **Candidate comparison view** — side-by-side emotion timelines for multiple
  candidates who answered the same question
- **Zoom/Teams webhook** — push report link to recruiter via webhook on call end

---

## Resources by phase

| Phase | Primary resource |
|---|---|
| 0 — Setup | FastAPI docs · MDN WebRTC guide |
| 1 — Classical CV | Szeliski Ch 3–5 (free PDF) · OpenCV docs |
| 2 — Classical ML | ISLR Ch 2–8 (free at statlearning.com) |
| 3 — CNN | fast.ai Lesson 1–4 · Goodfellow Ch 9 |
| 4 — LSTM | Goodfellow Ch 10 · Colah's LSTM blog post |
| 5 — Speech | Whisper paper · librosa docs |
| 6 — Fusion | ISLR Ch 8 (ensemble) · scikit-learn clustering guide |
| 7–8 — Frontend/Report | MDN WebRTC · Chart.js docs |

---