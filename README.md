# 🎙️ Real-Time Hindi, English & Hinglish Speech-to-Text

> **Implementation note:** the repository originally contained this design document only.
> The current implementation follows it with independent capture, segmentation, ASR,
> optional translation, and GUI stages. Configuration lives in `config.py`; start with
> `python main.py`.

A fast, real-time Speech Recognition desktop application built with **Python, PyQt6, SoundCard, and Faster-Whisper**.

The application continuously listens to microphone input and converts speech into text with a focus on **low latency, transcription accuracy, and Hindi/English/Hinglish support**.

It is designed to handle natural code-switching between Hindi and English while preserving technical terms such as Python, SQLAlchemy, FastAPI, Jenkins, Docker, GitHub, Jira, and Pull Request.

---

## ✨ Features

* 🎤 Real-time microphone listening
* ⚡ Low-latency speech-to-text
* 🇬🇧 English recognition
* 🇮🇳 Hindi recognition
* 🔀 Hinglish recognition
* 🌐 Automatic language detection
* 📝 Live partial transcription
* ✅ Stable final transcription
* 🔤 Optional Hindi transliteration
* 🌍 Optional translation
* 🧠 Context-aware transcription
* 💻 Technical vocabulary support
* 🔇 Speech/silence detection
* 🧹 Text cleanup and duplicate removal
* 📊 CPU and memory monitoring
* 🖥️ PyQt6 desktop interface
* 🧵 Multi-threaded audio/transcription pipeline
* 🚀 CPU and GPU support
* 📋 Application and performance logging

---

# 🏗️ Architecture

The application uses a non-blocking producer/consumer architecture.

```text
                    ┌──────────────────┐
                    │    Microphone    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ SoundCard Capture│
                    │      Thread      │
                    └────────┬─────────┘
                             │
                             ▼
                       Audio Queue
                             │
                             ▼
                    ┌──────────────────┐
                    │ Speech / Silence │
                    │    Detection     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Rolling Audio   │
                    │     Buffer       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Faster-Whisper   │
                    │    ASR Worker    │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
                    ▼                  ▼
             Partial Text         Final Text
                    │                  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Language / Text  │
                    │    Processing    │
                    └────────┬─────────┘
                             │
                             ▼
                       PyQt Signals
                             │
                             ▼
                    ┌──────────────────┐
                    │    PyQt6 GUI     │
                    └──────────────────┘
```

Audio recording, transcription, and GUI updates run independently so that Whisper inference does not stop microphone capture or freeze the interface.

---

# 🛠️ Technology Stack

| Technology            | Purpose                              |
| --------------------- | ------------------------------------ |
| Python                | Core application                     |
| PyQt6                 | Desktop GUI                          |
| SoundCard             | Real-time microphone capture         |
| NumPy                 | Audio processing                     |
| Faster-Whisper        | Speech recognition                   |
| CTranslate2           | Optimized Whisper inference          |
| Indic Transliteration | Hindi transliteration                |
| Transformers          | Translation support                  |
| MarianMT              | Optional machine translation         |
| psutil                | CPU/RAM monitoring                   |
| Threading             | Background workers                   |
| Queue                 | Thread-safe audio/task communication |

---

# 🌎 Supported Languages

## English

Example speech:

```text
I need to review the pull request today.
```

Output:

```text
I need to review the pull request today.
```

## Hindi

Example:

```text
मुझे आज ऑफिस जाना है
```

Output:

```text
मुझे आज ऑफिस जाना है।
```

## Hinglish

The application supports switching between Hindi and English within the same sentence.

Example speech:

```text
Today main SQLAlchemy upgrade task pe work kar raha hoon
and after that I'll create the pull request.
```

Expected output:

```text
Today main SQLAlchemy upgrade task pe work kar raha hoon,
and after that I'll create the pull request.
```

The application does not intentionally translate mixed-language speech into completely Hindi or completely English.

---

# ⚡ Real-Time Processing

Unlike traditional transcription systems:

```text
Record
   ↓
Stop Recording
   ↓
Transcribe
   ↓
Display
```

this application follows a streaming approach:

```text
Listen
  ↓
Detect Speech
  ↓
Create Short Audio Chunk
  ↓
Transcribe
  ↓
Display Partial Text
  ↓
Continue Listening
  ↓
Update Partial Text
  ↓
Detect Speech End
  ↓
Finalize Text
```

This significantly reduces perceived transcription latency.

---

# 📝 Partial & Final Transcription

While speaking, the application can display temporary transcription.

```text
Live:
I am currently working on the SQLAlchemy...
```

After the sentence becomes stable:

```text
Final:
I am currently working on the SQLAlchemy upgrade.
```

Partial results are replaced instead of repeatedly appended.

This prevents duplicated output such as:

```text
I am
I am working
I am working on
I am working on SQLAlchemy
```

---

# 🔀 Hinglish Processing

Hinglish support is one of the main goals of the project.

The speech recognition pipeline attempts to preserve natural code-switching.

For example:

```text
Kal main office jaunga and then I'll review the PR.
```

should remain mixed-language text rather than automatically becoming:

```text
Tomorrow I will go to the office and review the PR.
```

Translation is treated as an optional feature.

---

# 💻 Technical Vocabulary

The application is designed to preserve commonly used software-development terminology.

Examples include:

```text
Python
PyQt6
FastAPI
SQLAlchemy
Alembic
Pydantic
Jenkins
Docker
Kubernetes
Git
GitHub
GitLab
Jira
pytest
Kafka
Redis
MongoDB
PostgreSQL
API
REST API
Pull Request
PR
Commit
Branch
Pipeline
Database
```

Additional project-specific terminology can be added to the vocabulary/context configuration.

---

# 🔤 Transliteration

The project uses:

```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
```

Transliteration can be configured independently from transcription.

Possible modes:

```text
Original
Latin
Devanagari
```

Technical English terminology should remain unchanged whenever possible.

---

# 🌐 Translation

Optional translation is supported using:

```python
from transformers import MarianMTModel, MarianTokenizer
```

and MarianMT models.

Translation is intentionally kept outside the critical real-time transcription path.

```text
Speech
   ↓
Faster-Whisper
   ↓
Display Transcription
   ↓
Optional Translation
```

This prevents translation from delaying live speech recognition.

---

# 🧵 Threading Model

The application separates expensive operations into background workers.

```text
Main Thread
    │
    └── PyQt6 GUI

Capture Thread
    │
    └── Microphone Recording

Processing Thread
    │
    └── Speech Detection / Buffering

ASR Thread
    │
    └── Faster-Whisper Inference

Optional Worker
    │
    └── Translation / Post-processing
```

Communication between workers is handled using:

```python
Queue
Event
pyqtSignal
```

PyQt widgets are updated only from the main GUI thread.

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
soundcard
numpy
psutil
PyQt6
faster-whisper
indic-transliteration
transformers
torch
sentencepiece
```

Exact dependency versions should be maintained in `requirements.txt`.

## Running tests

From the `SpeakScribe` directory, run:

```bash
python -m pytest
```

The included `pytest.ini` explicitly adds the repository root to Python's import
path. `tests/conftest.py` applies the same setup before test collection for
Windows `pytest.exe`, IDE test runners, and wrappers that override pytest's
configuration. This keeps top-level application modules such as `audio_pipeline`
and `text_processing` importable with both `pytest` and `python -m pytest`.

`soundcard` is loaded only when microphone capture starts. Pure NumPy components
such as `EnergySpeechDetector` can therefore be imported and tested without an
installed or initialized platform audio backend. Running the application still
requires all packages from `requirements.txt`.

Likewise, `indic-transliteration` is loaded only when Latin or Devanagari script
conversion is selected. Language detection, cleanup, and the default `Original`
script mode do not require the optional transliteration package during import.

For quiet laptop microphone input, the default RMS speech threshold is `0.004`.
Accepted speech is DC-centered and receives capped gain before Whisper inference;
silence still remains excluded by the RMS detector. Detection logs include the
measured RMS value so the thresholds can be tuned for a particular microphone.

---

# 🚀 Running the Application

Run:

```bash
python main.py
```

The application will initialize the speech-recognition model and display the PyQt6 interface.

Click:

```text
Start Listening
```

and begin speaking.

Use:

```text
Stop Listening
```

to stop microphone processing safely.

---

# 🎛️ Configuration

Important settings should be maintained in a centralized configuration module.

Example:

```python
SAMPLE_RATE = 16000

MODEL_SIZE = "small"

DEVICE = "auto"
COMPUTE_TYPE = "auto"

PERFORMANCE_MODE = "balanced"

PARTIAL_INTERVAL = 0.7
SILENCE_DURATION = 0.5
MIN_SPEECH_DURATION = 0.2

SCRIPT_MODE = "original"

TRANSLATION_ENABLED = False

MAX_AUDIO_QUEUE = 100
MAX_ASR_QUEUE = 2

CONTEXT_SENTENCES = 2
```

---

# ⚙️ Performance Modes

The application can support multiple performance profiles.

### Fast

Prioritizes response time.

Suitable for:

* CPU systems
* Live dictation
* Lower-end hardware

### Balanced

Balances speed and accuracy.

Recommended default mode.

### Accurate

Prioritizes transcription accuracy.

Suitable when additional processing latency is acceptable.

---

# 🎮 CPU / GPU

Faster-Whisper supports optimized inference using CTranslate2.

## CPU

The application can run completely on CPU using an appropriate compute type.

## NVIDIA GPU

CUDA can significantly improve transcription speed when a compatible NVIDIA GPU and runtime are available.

The application should automatically fall back to CPU if GPU initialization fails.

---

# 📊 Performance Monitoring

The application can monitor its own resource usage using `psutil`.

Example metrics:

```text
Audio Duration:      1.20 sec
Whisper Inference:   0.31 sec
Real-Time Factor:    0.26
Display Latency:     420 ms
Audio Queue:         0
ASR Queue:           0
Memory:              1.8 GB
```

These metrics help identify transcription bottlenecks.

---

# 📋 Logging

The project uses its own logging utilities:

```python
from logger import log_print, get_output_path
```

Important events can be logged, including:

```text
Application startup
Model loading
Microphone initialization
Listening started
Speech detected
Speech ended
Partial transcription
Final transcription
Language detection
Inference duration
CPU/RAM usage
Warnings
Exceptions
Application shutdown
```

---

# 🔇 Silence & Hallucination Protection

Speech/silence detection helps prevent Whisper from processing unnecessary silent audio.

The transcription pipeline can additionally evaluate:

```text
No-speech probability
Minimum speech duration
Transcription confidence
Repeated output
Audio energy
```

Low-confidence or silence-generated results can be discarded.

The system should never intentionally generate sentences simply to make incomplete speech grammatically correct.

---

# 🎯 Performance Goal

Target user experience:

```text
Speak
  ↓
Speech detected
  ↓
~300–800 ms
  ↓
First useful partial transcription
  ↓
Continuous updates
  ↓
Short pause
  ↓
Stable final transcription
```

Actual latency depends on hardware, model size, audio quality, and configuration.

---

# 📁 Suggested Project Structure

```text
speech-to-text/
│
├── main.py
├── config.py
├── logger.py
├── requirements.txt
├── README.md
│
├── audio/
│   ├── __init__.py
│   ├── microphone.py
│   ├── speech_detector.py
│   └── audio_buffer.py
│
├── asr/
│   ├── __init__.py
│   ├── whisper_engine.py
│   └── language_detector.py
│
├── processing/
│   ├── __init__.py
│   ├── text_cleaner.py
│   ├── transliterator.py
│   └── translator.py
│
├── ui/
│   ├── __init__.py
│   └── main_window.py
│
└── tests/
    ├── test_language_detector.py
    ├── test_text_cleaner.py
    └── test_audio_buffer.py
```

The exact structure may differ depending on the existing implementation.

---

# 🔄 Processing Pipeline

```text
Microphone
    │
    ▼
Audio Capture
    │
    ▼
Speech Detection
    │
    ▼
Rolling Buffer
    │
    ├─────── Continue Recording
    │
    ▼
Faster-Whisper
    │
    ├── English
    ├── Hindi
    └── Hinglish
    │
    ▼
Text Cleanup
    │
    ├── Optional Transliteration
    └── Optional Translation
    │
    ▼
PyQt Signal
    │
    ▼
Live GUI
```

---

# 🛣️ Development Roadmap

### Phase 1 — Core Streaming

* Continuous SoundCard microphone capture
* Non-blocking audio queues
* Faster-Whisper integration
* PyQt thread-safe updates

### Phase 2 — Real-Time Recognition

* Speech/silence detection
* Rolling audio buffer
* Partial transcription
* Final transcription
* Duplicate prevention

### Phase 3 — Multilingual Support

* Hindi recognition
* English recognition
* Hinglish/code-switching
* Language detection
* Technical vocabulary

### Phase 4 — Text Processing

* Context preservation
* Text cleanup
* Optional transliteration
* Optional MarianMT translation

### Phase 5 — Optimization

* CPU optimization
* CUDA acceleration
* Queue optimization
* Memory optimization
* Latency monitoring
* Hallucination reduction

---

# 🧪 Example

### Speech

```text
Today main SQLAlchemy upgrade task pe work kar raha hoon
and after that I'll create the pull request.
```

### Live Output

```text
🎤 Listening...

Language: Hinglish

Live:
Today main SQLAlchemy upgrade task pe work...
```

### Final Output

```text
Today main SQLAlchemy upgrade task pe work kar raha hoon,
and after that I'll create the pull request.
```

---

# 🔮 Future Improvements

Potential future enhancements include:

* Advanced Voice Activity Detection
* Speaker identification
* Multiple microphone profiles
* Custom vocabulary management
* Meeting transcription
* Timestamped transcription
* Export to TXT/JSON/SRT
* Audio recording history
* Automatic meeting notes
* Improved Hinglish normalization
* Additional Indian languages
* Offline model management

---

# 📌 Project Goals

The project focuses on three primary goals:

**1. Speed**
Speech should appear as text as quickly as reasonably possible.

**2. Accuracy**
Hindi, English, Hinglish, and technical terminology should be recognized reliably.

**3. Stability**
Continuous listening should remain responsive without blocking the GUI, accumulating excessive queues, or consuming unnecessary resources.

---

## License

Add the appropriate project license here.

---

## Contributing

Contributions, bug reports, and performance improvements are welcome.

When making changes, preserve existing functionality and verify that modifications do not negatively affect transcription latency or accuracy.
