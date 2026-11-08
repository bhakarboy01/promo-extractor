import os
import re
import json
import importlib
import streamlit as st
from typing import List, Dict, Any, Tuple

# Hot-reload resilience for Streamlit Cloud
import llm_provider
if not hasattr(llm_provider, "LOG_FILE") or not hasattr(llm_provider, "LLM_CHATS_FILE"):
    llm_provider = importlib.reload(llm_provider)

from llm_provider import LiteLLMProvider, resolve_api_keys
LOG_FILE = getattr(llm_provider, "LOG_FILE", os.path.join(os.path.dirname(__file__), "outputs", "pipeline_run.log"))
LLM_CHATS_FILE = getattr(llm_provider, "LLM_CHATS_FILE", os.path.join(os.path.dirname(__file__), "outputs", "extras", "llm_chats.json"))

import promo_engine
if not hasattr(promo_engine, "PromoEngine"):
    promo_engine = importlib.reload(promo_engine)

from promo_engine import (
    PromoEngine,
    load_data_files,
    load_episodes_from_zip,
    extract_episode_number,
    build_xml_tagged_corpus,
    EXTRACTED_PROMOS_FILE,
    RANKED_PROMOS_FILE,
    XML_CORPUS_FILE
)

EXTRAS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "extras")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

def find_deliverable_path(filename_list: List[str]) -> str:
    for fname in filename_list:
        for folder in [DOCS_DIR, EXTRAS_DIR]:
            candidate = os.path.join(folder, fname)
            if os.path.exists(candidate):
                return candidate
    return os.path.join(EXTRAS_DIR, filename_list[0])

PROMPT_TEMPLATES_FILE = find_deliverable_path(["prompt_templates.md"])
TOUR_SCRIPT_FILE = find_deliverable_path(["presentation_tour_script.txt"])
MERMAID_FILE = find_deliverable_path(["architecture_flowchart.mmd"])
SOLUTION_DOC_MD = find_deliverable_path(["solution_doc.md"])
SOLUTION_DOC_TEX = find_deliverable_path(["solution_architecture_document.tex", "solution_doc.tex"])

st.set_page_config(
    page_title="AI Microdrama Promo Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8A00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #aaaaaa;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #2d3139;
        margin-bottom: 12px;
    }
    .promo-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 16px;
    }
    .hook-box {
        background: rgba(255, 186, 0, 0.12);
        border: 1px solid #FFBA00;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .cliff-box {
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid #FF4B4B;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .pairing-box {
        background: rgba(138, 43, 226, 0.12);
        border: 1px solid #8A2BE2;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        background: #2e3546;
        color: #eee;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-rank {
        background: #FF4B4B;
        color: white;
    }
    .badge-dur {
        background: #00A86B;
        color: white;
    }
    .badge-score {
        background: #8A2BE2;
        color: white;
    }
    .badge-relevance {
        background: #008080;
        color: white;
    }
    .badge-distant {
        background: #FF8A00;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/clapperboard.png", width=64)
    st.title("⚙️ Production Studio")

    st.markdown("""
    [![Live Cloud App](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://promo-extractor-hemant.streamlit.app/)
    """, unsafe_allow_html=True)
    st.caption("🌐 [promo-extractor-hemant.streamlit.app](https://promo-extractor-hemant.streamlit.app/)")

    st.markdown("""
    <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 6px; border: 1px solid #333; margin-top: 6px; margin-bottom: 12px;">
        <div style="font-size: 0.8rem; color: #aaa; margin-bottom: 2px;">Developed by:</div>
        <div style="font-weight: 700; color: #fff; font-size: 0.95rem; margin-bottom: 6px;">Hemant Bhakar</div>
        <div style="display: flex; gap: 12px;">
            <a href="https://github.com/bhakarboy01" target="_blank" style="text-decoration: none; color: #FF4B4B; font-size: 0.82rem; font-weight: 600;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="14" style="vertical-align: middle; filter: invert(1); margin-right: 3px;"/> GitHub
            </a>
            <a href="https://www.linkedin.com/in/bhakar-hemant/" target="_blank" style="text-decoration: none; color: #0077B5; font-size: 0.82rem; font-weight: 600;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-plain.svg" width="14" style="vertical-align: middle; margin-right: 3px;"/> LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ Evaluator API & Routing")
    st.caption("Custom LLM routing and evaluator key injection for recruiters and engineers.")

    default_env_keys = os.environ.get("GEMINI_API_KEY", "")

    custom_api_keys = st.text_input(
        "Inject Custom API Key(s) (Comma separated)",
        value=default_env_keys,
        type="password",
        placeholder="e.g. AIza..., sk-proj-..., sk-ant-...",
        help="Evaluator API keys are masked with password bullets for privacy. Supports Google Gemini, OpenAI, or Anthropic keys."
    )

    routing_mode = st.radio(
        "Routing Mode",
        ["Optimized Default (Free Tier)", "Global Override", "Advanced Task Routing"],
        index=0,
        help="Configure how models are dispatched across extraction stages."
    )

    MODEL_DROPDOWN_OPTIONS = [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20240620",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-2.0-flash",
        "Custom"
    ]

    selected_global_model = None
    stage_2_model = None
    stage_3_model = None
    stage_4_model = None

    if routing_mode == "Optimized Default (Free Tier)":
        st.info("💡 **Optimized Free-Tier Stack**:\n"
                "- **Stage 2 (Skim)**: `gemini-1.5-flash` (Cost-effective macro scanning)\n"
                "- **Stage 3 (Editor)**: `gemini-1.5-pro` (Deep reasoning & timestamp precision)\n"
                "- **Stage 4 (Ranker)**: `gemini-1.5-pro` (Cohesive marketing synergy ranking)\n\n"
                "*Automatic failover to active Google AI Studio flash-lite endpoints if limits are reached.*")

    elif routing_mode == "Global Override":
        selected_model_choice = st.selectbox(
            "Select Global Model Override",
            MODEL_DROPDOWN_OPTIONS,
            index=0,
            help="This model will be used across all stages (with fallback to the free-tier stack if it fails)."
        )
        if selected_model_choice == "Custom":
            selected_global_model = st.text_input("Enter Custom Model String (LiteLLM format)", "gpt-4o")
        else:
            selected_global_model = selected_model_choice

    elif routing_mode == "Advanced Task Routing":
        st.caption("Configure dedicated models per extraction stage:")
        
        # Stage 2: Skim
        s2_choice = st.selectbox("Stage 2: Macro Director Skim", MODEL_DROPDOWN_OPTIONS, index=4, help="Default: gemini/gemini-1.5-flash")
        stage_2_model = st.text_input("Custom Skim Model", "gemini/gemini-1.5-flash") if s2_choice == "Custom" else s2_choice

        # Stage 3: Editor
        s3_choice = st.selectbox("Stage 3: Precision Concept Editor", MODEL_DROPDOWN_OPTIONS, index=3, help="Default: gemini/gemini-1.5-pro")
        stage_3_model = st.text_input("Custom Editor Model", "gemini/gemini-1.5-pro") if s3_choice == "Custom" else s3_choice

        # Stage 4: Ranker
        s4_choice = st.selectbox("Stage 4: Executive Cohesion Ranker", MODEL_DROPDOWN_OPTIONS, index=3, help="Default: gemini/gemini-1.5-pro")
        stage_4_model = st.text_input("Custom Ranker Model", "gemini/gemini-1.5-pro") if s4_choice == "Custom" else s4_choice

    apply_btn = st.button("💾 Apply Configuration", type="secondary", use_container_width=True)

    # Parse entered keys
    parsed_eval_keys = [k.strip().strip('"').strip("'") for k in re.split(r"[\r\n,;]+", custom_api_keys) if k.strip()]

    # Initialize or update session state for routing
    if "routing_config" not in st.session_state:
        st.session_state["routing_config"] = {
            "mode": "Optimized Default (Free Tier)",
            "global_model": None,
            "stage_routes": {},
            "keys": parsed_eval_keys
        }

    if apply_btn:
        st.session_state["routing_config"] = {
            "mode": routing_mode,
            "global_model": selected_global_model if routing_mode == "Global Override" else None,
            "stage_routes": {
                "Macro Narrative Director Skim": stage_2_model,
                "Precision Concept Editor": stage_3_model,
                "Executive Cohesion Ranker": stage_4_model
            } if routing_mode == "Advanced Task Routing" else {},
            "keys": parsed_eval_keys
        }
        st.success(f"✅ Configuration applied! ({len(parsed_eval_keys)} active key(s) in pool)")

    active_cfg = st.session_state["routing_config"]
    if active_cfg["keys"]:
        st.caption(f"🛡️ **{len(active_cfg['keys'])} API Key(s)** active in failover pool.")
    else:
        st.warning("⚠️ No API keys configured.")

    st.subheader("⏱️ Rate-Limit & Pacing Control")
    pacing_delay = st.slider(
        "Pacing Sleep Between Requests (Seconds)",
        min_value=0.5,
        max_value=6.0,
        value=2.5,
        step=0.5,
        help="Adds a healthy sleep between requests to strictly prevent Gemini 15 RPM / 429 rate limit spikes."
    )

    temperature = st.slider("Sampling Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()

    st.subheader("📂 Screenplay Source")
    source_type = st.radio(
        "Ingestion Source",
        ["Use Existing Dataset (data/)", "Upload ZIP Archive (.zip)", "Upload Text Files (.txt)"],
        index=0
    )

    st.caption("🎯 **Quota Standards:** 15 Concepts ➔ Minimum 8–10 Valid Promos (Strictly 20–60s).")

# Main Title & Subtitle
st.markdown('<div class="main-header">AI Microdrama Promo Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Concept-Driven Pipeline: Macro Narrative Analysis ➔ Precision Timestamp Editing ➔ Cohesion & Virality Ranking</div>', unsafe_allow_html=True)

# Ingestion Flow
loaded_episodes: List[Tuple[int, str, str]] = []

if source_type == "Use Existing Dataset (data/)":
    loaded_episodes = load_data_files("data")
    if loaded_episodes:
        st.success(f"✅ Found **{len(loaded_episodes)} screenplay episodes** in local `data/` folder.")
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            episode_limit = st.slider(
                "Episodes Included in Analysis",
                min_value=5,
                max_value=len(loaded_episodes),
                value=min(20, len(loaded_episodes)),
                help="Select how many episodes from data/ to compile into the series corpus."
            )
        with col_s2:
            st.metric("Corpus Size", f"{episode_limit} Episodes")
        selected_episodes = loaded_episodes[:episode_limit]
    else:
        st.warning("No script files found in `data/`. Please upload a ZIP archive or files below.")
        selected_episodes = []

elif source_type == "Upload ZIP Archive (.zip)":
    uploaded_zip = st.file_uploader(
        "Upload Screenplay Episodes Archive (.zip)",
        type=["zip"],
        help="Upload a single ZIP archive containing your screenplay episode .txt files."
    )
    if uploaded_zip is not None:
        try:
            zip_bytes = uploaded_zip.read()
            loaded_episodes = load_episodes_from_zip(zip_bytes)
            st.success(f"✅ Successfully extracted and indexed **{len(loaded_episodes)} episodes** from `{uploaded_zip.name}`.")
            selected_episodes = loaded_episodes
        except Exception as e:
            st.error(f"❌ Could not unpack ZIP archive: {str(e)}")
            selected_episodes = []
    else:
        selected_episodes = []

else:
    uploaded_files = st.file_uploader("Upload Episode Scripts (.txt)", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            content = uf.read().decode("utf-8", errors="replace")
            ep_num = extract_episode_number(uf.name)
            loaded_episodes.append((ep_num, uf.name, content))
        loaded_episodes.sort(key=lambda x: x[0])
        st.success(f"✅ Uploaded **{len(loaded_episodes)} episodes**.")
        selected_episodes = loaded_episodes
    else:
        selected_episodes = []

# Action Button
col_btn, col_stats = st.columns([1, 3])
with col_btn:
    run_button = st.button("🚀 Run Extraction Pipeline", type="primary", use_container_width=True)

# Engine Initialization with Evaluator Dynamic Routing
active_cfg = st.session_state.get("routing_config", {
    "mode": "Optimized Default (Free Tier)",
    "global_model": None,
    "stage_routes": {},
    "keys": parsed_eval_keys
})

active_keys = active_cfg.get("keys") or parsed_eval_keys

llm_provider = LiteLLMProvider(
    api_key=active_keys if active_keys else None,
    temperature=temperature,
    pacing_delay=pacing_delay
)

# Apply routing methods based on evaluator selection
if active_cfg.get("mode") == "Global Override" and active_cfg.get("global_model"):
    llm_provider.set_global_override(active_cfg["global_model"])
elif active_cfg.get("mode") == "Advanced Task Routing" and active_cfg.get("stage_routes"):
    for stg, mdl in active_cfg["stage_routes"].items():
        llm_provider.set_custom_route(stg, mdl)
else:
    llm_provider.set_global_override(None)  # Enforce default optimized free-tier stack

engine = PromoEngine(
    llm_provider=llm_provider,
    pacing_delay=pacing_delay
)

# Pipeline Execution
if run_button:
    if not selected_episodes:
        st.error("Please provide screenplay episodes to process.")
    elif not active_keys:
        st.error("Please provide at least one API Key in the sidebar or in `.env`.")
    else:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def update_progress(msg: str, frac: float):
            progress_bar.progress(min(1.0, frac))
            status_box.info(msg)

        try:
            with st.spinner("Processing episodes through Concept-Driven pipeline..."):
                pipeline_result = engine.run_concept_driven_pipeline(
                    episodes=selected_episodes,
                    progress_callback=update_progress
                )

            st.session_state["pipeline_result"] = pipeline_result
            st.session_state["promo_concepts"] = pipeline_result["promo_concepts"]
            st.session_state["extracted_promos"] = pipeline_result["extracted_promos"]
            st.session_state["ranking_result"] = pipeline_result["ranking_result"]
            st.session_state["xml_corpus"] = pipeline_result["xml_corpus"]

            status_box.success(
                f"🎉 Pipeline Complete! Discovered {len(pipeline_result['promo_concepts'])} viral concepts, "
                f"successfully extracted {len(pipeline_result['extracted_promos'])} promos under 60s, "
                f"and ranked all {len(pipeline_result['ranking_result'].get('top_promos', []))} promos in outputs/ folder."
            )
        except Exception as e:
            status_box.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)

# Load existing deliverables from outputs/ folder if present
if os.path.exists(EXTRACTED_PROMOS_FILE) and "extracted_promos" not in st.session_state:
    try:
        with open(EXTRACTED_PROMOS_FILE, "r", encoding="utf-8") as f:
            st.session_state["extracted_promos"] = json.load(f)
    except Exception:
        pass

if os.path.exists(RANKED_PROMOS_FILE) and "ranking_result" not in st.session_state:
    try:
        with open(RANKED_PROMOS_FILE, "r", encoding="utf-8") as f:
            st.session_state["ranking_result"] = json.load(f)
    except Exception:
        pass

# Render Tabs with clean professional headings
tab_ranked, tab_concepts, tab_deliverables, tab_xml, tab_logs = st.tabs([
    "🏆 Top Ranked Viral Promos",
    "💡 Clip Concepts & Editorial Status",
    "📜 Master Deliverables (JSON Exports)",
    "📑 Screenplay Corpus Inspector",
    "📋 Live Pipeline Run Logs"
])

# TAB 1: TOP RANKED VIRAL PROMOS (SHOW TOP 7 AS REQUESTED)
with tab_ranked:
    st.subheader("🏆 Top Ranked Viral Promotional Clips")
    st.markdown("Evaluated as a **single cohesive unit**, prioritizing the psychological pairing between the opening hook and closing cliffhanger.")

    ranking_data = st.session_state.get("ranking_result")
    if ranking_data and "top_promos" in ranking_data and ranking_data["top_promos"]:
        all_ranked_promos = ranking_data["top_promos"]
        top_7_promos = all_ranked_promos[:7]
        remaining_promos = all_ranked_promos[7:]

        if "ranking_summary" in ranking_data and ranking_data["ranking_summary"]:
            st.info(f"💡 **Executive Producer Strategy:** {ranking_data['ranking_summary']}")

        st.markdown(f"#### 🌟 Featured Selection: Top 7 Ranked Cuts (of {len(all_ranked_promos)} Total Ranked)")

        def render_promo_card(item: Dict[str, Any]):
            rank = item.get("rank", 1)
            score = item.get("virality_score", 92)
            dur = item.get("total_duration_seconds", 45)
            eps = item.get("episodes_used", [])
            title = item.get("title", f"Promotional Cut #{rank}")
            hook = item.get("hook", "")
            cliff = item.get("cliffhanger", "")
            relevance = item.get("relevance", "High Virality & Retention")
            pairing = item.get("pairing_analysis", "")
            timestamps = item.get("timestamps", [])
            why = item.get("why_it_works", "")

            is_distant = len(eps) > 1 and max(eps) - min(eps) > 3

            with st.container():
                badges_html = (
                    f'<div class="promo-card">'
                    f'<span class="badge badge-rank">Rank #{rank}</span>'
                    f'<span class="badge badge-score">Virality Score: {score}/100</span>'
                    f'<span class="badge badge-relevance">🎯 {relevance}</span>'
                    f'<span class="badge badge-dur">⏱️ {dur}s (Strictly 20-60s)</span>'
                    f'<span class="badge">Episodes: {", ".join(str(e) for e in eps)}</span>'
                    f'{"<span class=\"badge badge-distant\">⚡ Cross-Episode Spliced</span>" if is_distant else ""}'
                    f'<div style="font-size: 1.45rem; font-weight: 700; color: #FFFFFF; margin-top: 10px; margin-bottom: 4px; letter-spacing: -0.01em;">{title}</div>'
                    f'</div>'
                )
                st.markdown(badges_html, unsafe_allow_html=True)

                col_a, col_b = st.columns([3, 2])
                with col_a:
                    boxes_html = (
                        f'<div class="hook-box">'
                        f'<strong>🎣 The Hook (0-5s):</strong><br/>{hook}'
                        f'</div>'
                        f'<div class="cliff-box">'
                        f'<strong>⚡ The Cliffhanger:</strong><br/>{cliff}'
                        f'</div>'
                    )
                    st.markdown(boxes_html, unsafe_allow_html=True)
                    if pairing:
                        pairing_html = (
                            f'<div class="pairing-box">'
                            f'<strong>🔗 Hook & Cliffhanger Pairing Synergy:</strong><br/>{pairing}'
                            f'</div>'
                        )
                        st.markdown(pairing_html, unsafe_allow_html=True)
                    if why:
                        st.caption(f"🎯 **Why this works:** {why}")

                with col_b:
                    st.write("**🎞️ Exact Scene Timestamps:**")
                    ts_rows = []
                    for seg in timestamps:
                        ts_rows.append({
                            "Episode": f"Ep {seg.get('episode', eps[0] if eps else 'N/A')}",
                            "Start": seg.get("start", "00:00.00"),
                            "End": seg.get("end", "00:45.00")
                        })
                    st.dataframe(ts_rows, use_container_width=True, hide_index=True)

                st.divider()

        # Render Top 7
        for item in top_7_promos:
            render_promo_card(item)

        # Remaining Ranks (8 to 15) in clean expandable container
        if remaining_promos:
            with st.expander(f"🔽 View Remaining Ranked Promos (Ranks 8 to {len(all_ranked_promos)})"):
                st.caption(f"All {len(all_ranked_promos)} candidate promos have been fully ranked and saved to `outputs/ranked_promos.json`.")
                for item in remaining_promos:
                    render_promo_card(item)
    else:
        st.info("No ranked promos generated yet. Click **'Run Extraction Pipeline'** to execute.")

# TAB 2: CLIP CONCEPTS & EDITORIAL STATUS
with tab_concepts:
    st.subheader("💡 Viral Clip Concepts & Editorial Status")
    st.markdown("Macro narrative concepts discovered across the series and edited with targeted episode scripts.")

    concepts = st.session_state.get("promo_concepts", [])
    extracted_map = {p["promo_id"]: p for p in st.session_state.get("extracted_promos", [])}

    if concepts:
        st.write(f"Total Concepts Identified: **{len(concepts)}** | Successfully Extracted under 60s: **{len(extracted_map)}**")
        for c in concepts:
            c_id = c["concept_id"]
            eps = c["episodes_involved"]
            is_extracted = c_id in extracted_map
            status_badge = "✅ Extracted (20-60s)" if is_extracted else "⚠️ Omitted (Exceeded 60s)"
            is_distant = len(eps) > 1 and max(eps) - min(eps) > 3

            with st.expander(f"🎬 {c_id} - Episodes {eps} - {status_badge} {'[Cross-Episode Spliced]' if is_distant else ''}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Target Hook:**\n> {c['target_hook']}")
                    st.markdown(f"**Target Cliffhanger:**\n> {c['target_cliffhanger']}")
                with col2:
                    if is_extracted:
                        p_data = extracted_map[c_id]
                        st.success(f"**Total Duration:** {p_data.get('total_duration_seconds', 45)} seconds")
                        st.write("**Timestamps:**")
                        st.json(p_data.get("timestamps", []))
                        st.caption(f"**Editorial Rationale:** {p_data.get('why_it_works', '')}")
                    else:
                        st.warning("Omitted during precision editing to protect the strict 20-60s duration limit.")
    else:
        st.info("No concepts generated yet.")

# TAB 3: MASTER DELIVERABLES (JSON EXPORTS)
with tab_deliverables:
    st.subheader("📜 Master Deliverables (JSON Exports in `outputs/`)")
    st.markdown("Download and inspect the separate deliverable files stored in the `outputs/` folder.")

    col_deliv1, col_deliv2 = st.columns(2)

    with col_deliv1:
        st.markdown("### 1. `outputs/ranked_promos.json`")
        st.caption("All 15 candidate promotional clips evaluated and ranked as cohesive units with ranks and relevance.")
        if os.path.exists(RANKED_PROMOS_FILE):
            with open(RANKED_PROMOS_FILE, "r", encoding="utf-8") as f:
                ranked_json_str = f.read()
            st.download_button(
                label="📥 Download ranked_promos.json (All 15 Ranked)",
                data=ranked_json_str,
                file_name="ranked_promos.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )
            try:
                st.json(json.loads(ranked_json_str))
            except Exception:
                st.text(ranked_json_str)
        else:
            st.info("`outputs/ranked_promos.json` will be generated after running the pipeline.")

    with col_deliv2:
        st.markdown("### 2. `outputs/extracted_promos.json`")
        st.caption("Complete pool of candidate promotional clips extracted under the 20-60s duration constraint.")
        if os.path.exists(EXTRACTED_PROMOS_FILE):
            with open(EXTRACTED_PROMOS_FILE, "r", encoding="utf-8") as f:
                extracted_json_str = f.read()
            st.download_button(
                label="📥 Download extracted_promos.json",
                data=extracted_json_str,
                file_name="extracted_promos.json",
                mime="application/json",
                type="secondary",
                use_container_width=True
            )
            try:
                st.json(json.loads(extracted_json_str))
            except Exception:
                st.text(extracted_json_str)
        else:
            st.info("`outputs/extracted_promos.json` will be generated after running the pipeline.")

    st.divider()
    st.subheader("📁 Supplemental Deliverables (`outputs/extras/`)")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.markdown("### 3. `outputs/extras/llm_chats.json`")
        st.caption("Complete, single master log of ALL requests (prompts sent) and corresponding outputs received from the LLMs across all stages.")
        if os.path.exists(LLM_CHATS_FILE):
            with open(LLM_CHATS_FILE, "r", encoding="utf-8") as f:
                chats_json_str = f.read()
            try:
                chats_data = json.loads(chats_json_str)
                chat_count = chats_data.get("total_recorded_requests", len(chats_data.get("interactions", [])))
            except Exception:
                chat_count = "All"
            st.download_button(
                label=f"📥 Download llm_chats.json ({chat_count} Recorded Chats)",
                data=chats_json_str,
                file_name="llm_chats.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )
            with st.expander(f"🔍 Inspect Master LLM Chats File ({chat_count} Interactions)"):
                try:
                    st.json(json.loads(chats_json_str))
                except Exception:
                    st.text(chats_json_str)
        else:
            st.info("`outputs/extras/llm_chats.json` will be generated after running the pipeline.")

        st.markdown("### 4. `outputs/extras/prompt_templates.md`")
        st.caption("Comprehensive catalog of all system and user prompt templates with strategic rationale and JSON schemas.")
        if os.path.exists(PROMPT_TEMPLATES_FILE):
            with open(PROMPT_TEMPLATES_FILE, "r", encoding="utf-8") as f:
                pt_str = f.read()
            st.download_button(
                label="📥 Download prompt_templates.md",
                data=pt_str,
                file_name="prompt_templates.md",
                mime="text/markdown",
                use_container_width=True
            )
            with st.expander("🔍 View Prompt Templates Catalog"):
                st.markdown(pt_str)

    with col_ex2:
        st.markdown("### 5. `outputs/extras/series_screenplay_corpus.xml`")
        st.caption("Unified XML screenplay corpus containing all 50 episodes with episode tagging for macro narrative skimming.")
        if os.path.exists(XML_CORPUS_FILE):
            with open(XML_CORPUS_FILE, "r", encoding="utf-8") as f:
                xml_corpus_data = f.read()
            st.download_button(
                label=f"📥 Download series_screenplay_corpus.xml ({len(xml_corpus_data):,} chars)",
                data=xml_corpus_data,
                file_name="series_screenplay_corpus.xml",
                mime="application/xml",
                use_container_width=True
            )
        else:
            st.info("`outputs/extras/series_screenplay_corpus.xml` will be saved during ingestion.")

        st.markdown("### 6. `outputs/extras/presentation_tour_script.txt`")
        st.caption("Concise 2.5-minute spoken presentation script with explicit cues for switching between the architecture diagram and web app.")
        if os.path.exists(TOUR_SCRIPT_FILE):
            with open(TOUR_SCRIPT_FILE, "r", encoding="utf-8") as f:
                tour_script_text = f.read()
            st.download_button(
                label="📥 Download presentation_tour_script.txt",
                data=tour_script_text,
                file_name="presentation_tour_script.txt",
                mime="text/plain",
                use_container_width=True
            )
            with st.expander("🔍 Read 2.5-Minute Tour Script"):
                st.text(tour_script_text)

        st.markdown("### 7. `outputs/extras/architecture_flowchart.mmd`")
        st.caption("Mermaid flowchart script depicting data ingestion, LLM call sequence, and Map-Reduce promo synthesis.")
        if os.path.exists(MERMAID_FILE):
            with open(MERMAID_FILE, "r", encoding="utf-8") as f:
                mermaid_text = f.read()
            st.download_button(
                label="📥 Download architecture_flowchart.mmd",
                data=mermaid_text,
                file_name="architecture_flowchart.mmd",
                mime="text/plain",
                use_container_width=True
            )

    st.divider()
    st.subheader("📑 Formal Engineering Solution Documents (`docs/` & `outputs/extras/`)")
    col_sol1, col_sol2 = st.columns(2)
    with col_sol1:
        st.markdown("### 8. `outputs/extras/solution_doc.md`")
        st.caption("2-page engineering solution document covering problem framing, failed approach post-mortem, Map-Reduce pipeline, limitations, and scaling.")
        if os.path.exists(SOLUTION_DOC_MD):
            with open(SOLUTION_DOC_MD, "r", encoding="utf-8") as f:
                sol_md_text = f.read()
            st.download_button(
                label="📥 Download solution_doc.md",
                data=sol_md_text,
                file_name="solution_doc.md",
                mime="text/markdown",
                use_container_width=True
            )
            with st.expander("🔍 Read Solution Document (Markdown)"):
                st.markdown(sol_md_text)

    with col_sol2:
        st.markdown("### 9. `outputs/extras/solution_doc.tex` (Overleaf Ready)")
        st.caption("Publication-grade, Overleaf-ready LaTeX script strictly budgeted for a 2-page concise PDF submission.")
        if os.path.exists(SOLUTION_DOC_TEX):
            with open(SOLUTION_DOC_TEX, "r", encoding="utf-8") as f:
                sol_tex_text = f.read()
            st.download_button(
                label="📥 Download solution_doc.tex",
                data=sol_tex_text,
                file_name="solution_doc.tex",
                mime="text/x-tex",
                use_container_width=True
            )
            with st.expander("🔍 View Overleaf LaTeX Source"):
                st.code(sol_tex_text, language="latex")

# TAB 4: XML SCREENPLAY CORPUS INSPECTOR
with tab_xml:
    st.subheader("📑 Screenplay Corpus Inspector")
    st.markdown("The complete series screenplay compiled into a unified XML document, enabling macro narrative analysis across all episodes.")

    corpus_text = st.session_state.get("xml_corpus", "")
    if not corpus_text and selected_episodes:
        corpus_text = build_xml_tagged_corpus(selected_episodes)

    if corpus_text:
        col_xml_head, col_xml_dl = st.columns([2, 1])
        with col_xml_head:
            st.caption(f"Corpus Length: {len(corpus_text):,} characters (~{len(corpus_text)//4:,} tokens) | Saved to `outputs/extras/series_screenplay_corpus.xml`")
        with col_xml_dl:
            st.download_button(
                label="📥 Download Full XML Corpus",
                data=corpus_text,
                file_name="series_screenplay_corpus.xml",
                mime="application/xml",
                use_container_width=True
            )
        st.text_area("XML Corpus Preview", value=corpus_text[:12000] + ("\n\n... [truncated for preview] ..." if len(corpus_text) > 12000 else ""), height=450)
    else:
        st.info("No XML corpus available yet.")

# TAB 5: LIVE PIPELINE RUN LOGS
with tab_logs:
    st.subheader("📋 Pipeline Execution & Failover Logs (`outputs/pipeline_run.log`)")
    st.markdown("Live execution audit trail tracking key rotation, pacing delays, 429/503 error recovery, and LLM latency.")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
        col_log_dl, col_log_info = st.columns([1, 3])
        with col_log_dl:
            st.download_button(
                label="📥 Download pipeline_run.log",
                data=log_content,
                file_name="pipeline_run.log",
                mime="text/plain",
                use_container_width=True
            )
        with col_log_info:
            st.caption(f"Log Size: {len(log_content):,} bytes")
        st.text_area("Live Log Output", value=log_content, height=450)
    else:
        st.info("`outputs/pipeline_run.log` will be populated during pipeline execution.")

# Page Footer
st.markdown("""
<div style="text-align: center; padding: 25px 0 10px 0; border-top: 1px solid #2d3139; margin-top: 40px; color: #888;">
    <span style="font-weight: 600; color: #eee; font-size: 0.95rem;">AI Microdrama Promo Extractor</span> 
    <span style="color: #666; margin: 0 8px;">•</span> 
    Built by <strong style="color: #fff;">Hemant Bhakar</strong>
    <div style="margin-top: 8px;">
        <a href="https://github.com/bhakarboy01" target="_blank" style="margin-right: 18px; text-decoration: none; color: #FF4B4B; font-weight: 500; font-size: 0.9rem;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="16" style="vertical-align: middle; filter: invert(1); margin-right: 4px;"/> GitHub (@bhakarboy01)
        </a>
        <a href="https://www.linkedin.com/in/bhakar-hemant/" target="_blank" style="text-decoration: none; color: #0077B5; font-weight: 500; font-size: 0.9rem;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-plain.svg" width="16" style="vertical-align: middle; margin-right: 4px;"/> LinkedIn (bhakar-hemant)
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
