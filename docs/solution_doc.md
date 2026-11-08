# Context-Aware Microdrama Promo Extractor
## Engineering Solution Document & Technical Architecture Review
**Author:** Hemant Bhakar ([GitHub](https://github.com/bhakarboy01) | [LinkedIn](https://www.linkedin.com/in/bhakar-hemant/))  
**Live Deployed Application:** [promo-extractor-hemant.streamlit.app](https://promo-extractor-hemant.streamlit.app/)  
**Core Stack:** Python 3.10+, LiteLLM, Google Gemini 1.5 Flash/Pro, Streamlit Cloud

---

### 1. Executive Summary & Problem Framing
Short-form vertical microdramas (1–2 minute episodes, 50+ episodes per title) operate on extreme narrative acceleration. Driving paid viewer acquisition and organic TikTok/Reels virality requires high-octane **20–60 second promo trailers**. An effective promo cannot simply be a random excerpt; it requires:
1. **An Immediate Hook (0:00–0:05):** A visceral opening image or explosive dialogue beat that halts infinite vertical scrolling.
2. **A High-Stakes Cliffhanger (Final 5s):** An unresolved revelation or physical crisis compelling the viewer to install the app or unlock the next episode.
3. **Cross-Episode Non-Linear Splicing:** The most powerful narrative arcs span across distant episodes—e.g., establishing a subtle betrayal setup in Episode 2 and abruptly colliding it with a violent confrontation in Episode 15.

Standard naive single-episode summarization completely fails this requirement because it treats episodes as disconnected silos. Our objective was to build an autonomous, production-grade extraction engine that consumes an uncurated 50-episode screenplay corpus and outputs 15 broadcast-ready, time-stamped promo edit decision lists (EDLs) strictly clamped between 20 and 60 seconds.

---

### 2. Post-Mortem: Initial Failed Approaches
Before arriving at our final architecture, two standard LLM processing patterns were prototyped and subsequently abandoned due to fundamental narrative breakdown:

| Attempted Architecture | Core Operational Flaw | Why It Failed in Practice |
| :--- | :--- | :--- |
| **Attempt 1: Stateful Rolling Memory (`storyline_memory.txt`)** | Sequential processing loop passing Episode $N$ script + rolling summary string of Episodes $1 \dots (N-1)$ to the LLM. | **Progressive Narrative Drift & Amnesia:** As the episode count climbed past Ep 20, the rolling text buffer underwent severe compression. Subtle narrative seeds planted in early episodes (e.g., a hidden will in Ep 2) were discarded by the LLM as "minor background detail," making it impossible to pair with payoffs in Ep 45. |
| **Attempt 2: Fixed-Window Triplet Chunking** | Segmenting the series into rigid sequential chunks (`[Ep 1-3]`, `[Ep 4-6]`, etc.) and extracting local promos within each triplet. | **Artificial Narrative Isolation:** High-performing microdrama trailers thrive on wide chronological disparity (e.g., an innocent wedding vow in Ep 1 paired with an arrest warrant in Ep 18). Triplet chunking strictly forbade cross-window associations, yielding repetitive, low-tension micro-summaries. |

---

### 3. Final Production Architecture: Concept-Driven Map-Reduce
To resolve these bottlenecks, we engineered a three-tier **Concept-Driven Map-Reduce Pipeline** that separates macro-narrative strategy from micro-temporal editing:

1. **Stage 1 — Macro Narrative Director Skim:** The 50 individual screenplay episodes are normalized and bundled into a structured, lightweight XML corpus (`series_screenplay_corpus.xml`, 549 KB). The entire XML is dispatched in a single call to Gemini 1.5 Flash (leveraging its 1M+ token context window). The model operates as an executive creative director, scanning the complete macro narrative to produce exactly 10–15 viral promo concepts defining the dramatic premise, target hook, target cliffhanger, and exact episodes involved (single-episode or non-linear multi-episode pairs).
2. **Stage 2 — Precision Concept Editor:** Instead of re-passing the entire series, an isolated extraction agent iterates over each concept. It fetches **only the script text of the specific episodes involved** (e.g., retrieving exclusively Ep 2 and Ep 15). The LLM is instructed to identify exact cut-in and cut-out timestamps, character dialogue subtitles, and visual stage directions. It enforces a hard temporal boundary: total duration must strictly fall between **20.0 and 60.0 seconds**.
3. **Stage 3 — Executive Cohesion Ranker:** The complete set of 15 candidate promos is passed to an evaluation agent. The ranker evaluates narrative whiplash (emotional delta between hook and climax), pacing viability, and organic cliffhanger tension, sorting all 15 promos from Rank 1 to 15 and delivering an editorial justification for the Top 7.

---

### 4. Key Engineering Optimizations & Fault Tolerance
1. **8-Key Autonomous Failover Pool (`LiteLLMProvider`):** Free-tier AI Studio keys suffer from strict rate limits (15 RPM / 429 Resource Exhausted) and transient 503 gateway drops. We built an automatic round-robin failover pool supporting up to 8 API keys. When an active key triggers a rate limit or HTTP error, the engine seamlessly rotates to the next key without crashing or restarting the pipeline.
2. **Deterministic Pacing Delay:** A calibrated 2.5-second pacing sleep is executed between sequential LLM calls, ensuring steady throughput below per-minute tier thresholds.
3. **Dynamic Model Routing:**
   - **Default Free-Tier Stack:** Optimizes compute cost by assigning fast, high-context `gemini/gemini-1.5-flash` to the Macro Skim, and high-reasoning `gemini/gemini-1.5-pro` to the Precision Editor and Cohesion Ranker.
   - **Recruiter Custom Overrides:** Testing reviewers can supply custom models (OpenAI `gpt-4o`, Anthropic `claude-3-5-sonnet`, Groq `llama-3.3-70b-versatile`) either globally or per-stage. If a custom model fails or depletes its key, the provider falls back dynamically to the default free-tier stack.
4. **Autonomous JSON Schema Repair:** All LLM prompts enforce strict JSON schemas via system instructions and regex fallback cleaners, repairing truncated markdown code blocks or trailing commas automatically.

---

### 5. Quality Analysis: Honest Limitations & Failure Modes
| Failure Mode | Root Cause in Text Screenplays | Practical Mitigation / Future Fix |
| :--- | :--- | :--- |
| **1. The "Subtle Acting Blindspot"** | The LLM evaluates text dialogue and parentheticals. It cannot perceive an actor's lingering micro-expression, eye glance, or heavy silence that carries immense visual tension on camera. | Multimodal integration: Ingesting keyframe video embeddings and visual facial emotion classification alongside text. |
| **2. Audio Pacing & Hard Dialogue Cuts** | A script timestamp cut at `00:32` may sever an ongoing sentence cadence or collide with background orchestral music cues. | Audio waveform alignment: Incorporating Whisper word-level timestamps and silence detection before final cut locking. |
| **3. False Dramatic Cliffhangers** | The model occasionally flags mundane conversational interruptions (e.g., a phone ringing or door opening) as high-stakes narrative cliffhangers. | Enhanced prompt calibration with negative examples distinguishing mundane scene transitions from life-or-death suspense. |
| **4. Multi-Episode Visual Incongruity** | Splicing Episode 2 with Episode 25 may feature jarring character costume, hairstyle, or lighting shifts that break continuity in a trailer. | Character visual state tracking: Parsing wardrobe and scene tags across episodes to warn editors of high visual dissonance. |

---

### 6. Production Scaling to 500+ Shows & Unit Economics
- **Unit Economics per Show:** Processing an entire 50-episode microdrama (549 KB screenplay text) consumes ~145,000 input tokens and ~8,500 output tokens. On Gemini 1.5 Flash, the cost is **less than $0.035 per title**. Across an entire catalog of 500 shows, total extraction compute is under **$20.00 USD**.
- **Distributed Horizontal Scaling:** The Stage 2 Precision Editor executes independent extractions per concept. In production, this Map phase can be dispatched asynchronously across an asynchronous worker pool (Celery / AWS SQS / Redis), reducing end-to-end processing time from 3 minutes to **under 25 seconds per show**.
- **Automated Video Assembly:** The generated JSON EDLs cleanly output start/end timestamps and subtitle dialogue, ready for direct ingestion by headless FFmpeg renderers or programmatic timeline engines (e.g., Remotion, Shotstack).

---

### 7. AI Tool Transparency & Lessons Learned
- **Tools Leveraged:** Antigravity IDE, LiteLLM unified abstraction layer, Google Gemini 1.5 Flash/Pro models, Streamlit web framework.
- **Where AI Erred During Development:**
  1. *Model Versioning Drift:* AI Studio transitioned model endpoints, causing silent 404 errors when calling deprecated `gemini-1.5-flash` aliases under `v1beta`; resolved by standardizing on LiteLLM provider prefixes.
  2. *Markdown Indentation Bug:* Streamlit's CommonMark parser misinterpreted HTML strings with >= 4 space indents as raw code blocks, rendering literal HTML tags (`<h3 style="...">`) in the UI; resolved by unindenting markup and wrapping in styled `<div>` blocks.
  3. *Corpus Zero-Byte File Bug:* An unhandled default argument in the corpus builder wiped the disk corpus when invoked without arguments; resolved by decoupling in-memory string generation from disk serialization.

