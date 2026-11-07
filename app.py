import os
import json
import streamlit as st
from typing import List, Dict, Any, Tuple
from llm_provider import LiteLLMProvider, resolve_api_keys
from promo_engine import (
    PromoEngine,
    load_data_files,
    load_episodes_from_zip,
    extract_episode_number,
    build_xml_tagged_corpus,
    EXTRACTED_PROMOS_FILE,
    RANKED_PROMOS_FILE
)

st.set_page_config(
    page_title="AI Microdrama Promo Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8A00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #999999;
        margin-bottom: 1.5rem;
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
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        background: #2e3546;
        color: #eee;
        margin-right: 6px;
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
    st.title("⚙️ Production Controls")

    st.subheader("🤖 AI Model Engine")
    model_choice = st.selectbox(
        "Select Model (LiteLLM)",
        [
            "gemini-1.5-flash",
            "gemini-flash-latest",
            "gemini-3.1-flash-lite",
            "gemini-2.0-flash",
            "gpt-4o-mini",
            "claude-3-5-sonnet",
            "Custom"
        ],
        index=0,
        help="Default model: gemini-1.5-flash (mapped to active high-speed Google AI Studio endpoint)"
    )

    if model_choice == "Custom":
        active_model = st.text_input("Custom Model Name", "gemini-1.5-flash")
    else:
        active_model = model_choice

    env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "API Key (Gemini/OpenAI)",
        value=env_key,
        type="password",
        help="Prefilled from environment or enter key manually"
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input.strip()
        os.environ["GOOGLE_API_KEY"] = api_key_input.strip()

    temperature = st.slider("Sampling Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()

    st.subheader("📂 Screenplay Input")
    source_type = st.radio(
        "Ingestion Method",
        ["Use Existing Dataset (data/)", "Upload ZIP Archive (.zip)", "Upload Text Files (.txt)"],
        index=0
    )

    st.caption("🎯 **Quota Standards:** 10–15 Concepts ➔ Minimum 8–10 Valid Promos (Strictly 20–60s).")

# Main Title & Subtitle
st.markdown('<div class="main-header">AI Microdrama Promo Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Concept-Driven Pipeline: Macro Narrative Analysis ➔ Precision Timestamp Editing ➔ Cohesion & Virality Ranking</div>', unsafe_allow_html=True)

# Load Episodes based on selection
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

# Engine Initialization
llm_provider = LiteLLMProvider(
    model=active_model,
    api_key=api_key_input if api_key_input else None,
    temperature=temperature
)
engine = PromoEngine(llm_provider=llm_provider)

# Pipeline Execution
if run_button:
    if not selected_episodes:
        st.error("Please provide screenplay episodes to process.")
    elif not (api_key_input or os.environ.get("GEMINI_API_KEY")):
        st.error("Please provide a Gemini API Key in the sidebar or in `.env`.")
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
                f"🎉 Pipeline Complete! Generated {len(pipeline_result['promo_concepts'])} viral concepts, "
                f"successfully extracted {len(pipeline_result['extracted_promos'])} promos under 60s, "
                f"and generated ranked deliverables."
            )
        except Exception as e:
            status_box.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)

# Load existing deliverables if present on disk
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
tab_ranked, tab_concepts, tab_raw, tab_xml = st.tabs([
    "🏆 Top Ranked Viral Promos",
    "💡 Clip Concepts & Editorial Status",
    "📜 Master Deliverables (JSON Exports)",
    "📑 Screenplay Corpus Inspector"
])

# TAB 1: TOP RANKED VIRAL PROMOS
with tab_ranked:
    st.subheader("🏆 Top Ranked Viral Promotional Clips")
    st.markdown("Evaluated as a **single cohesive unit**, prioritizing the psychological pairing between the opening hook and closing cliffhanger.")
    
    ranking_data = st.session_state.get("ranking_result")
    if ranking_data and "top_promos" in ranking_data and ranking_data["top_promos"]:
        if "ranking_summary" in ranking_data and ranking_data["ranking_summary"]:
            st.info(f"💡 **Executive Producer Strategy:** {ranking_data['ranking_summary']}")

        for item in ranking_data["top_promos"]:
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
                st.markdown(f"""
                <div class="promo-card">
                    <span class="badge badge-rank">Rank #{rank}</span>
                    <span class="badge badge-score">Virality Score: {score}/100</span>
                    <span class="badge badge-relevance">🎯 {relevance}</span>
                    <span class="badge badge-dur">⏱️ {dur}s (Strictly 20-60s)</span>
                    <span class="badge">Episodes: {', '.join(str(e) for e in eps)}</span>
                    {'<span class="badge badge-distant">⚡ Cross-Episode Spliced</span>' if is_distant else ''}
                    <h3 style="margin-top:10px; margin-bottom: 5px;">{title}</h3>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b = st.columns([3, 2])
                with col_a:
                    st.markdown(f"""
                    <div class="hook-box">
                        <strong>🎣 The Hook (0-5s):</strong><br/>{hook}
                    </div>
                    <div class="cliff-box">
                        <strong>⚡ The Cliffhanger:</strong><br/>{cliff}
                    </div>
                    """, unsafe_allow_html=True)
                    if pairing:
                        st.markdown(f"""
                        <div class="pairing-box">
                            <strong>🔗 Hook & Cliffhanger Pairing Synergy:</strong><br/>{pairing}
                        </div>
                        """, unsafe_allow_html=True)
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
    else:
        st.info("No ranked promos generated yet. Click **'Run Extraction Pipeline'** to execute.")

# TAB 2: CLIP CONCEPTS & EDITORIAL STATUS
with tab_concepts:
    st.subheader("💡 Viral Clip Concepts & Editorial Status")
    st.markdown("Concepts identified during macro narrative analysis and edited with targeted episode scripts.")

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
with tab_raw:
    st.subheader("📜 Master Deliverables (JSON Exports)")
    st.markdown("Download and inspect the separate deliverables generated by the pipeline.")

    col_deliv1, col_deliv2 = st.columns(2)

    with col_deliv1:
        st.markdown("### 1. `ranked_promos.json`")
        st.caption("Top 5 ranked promotional clips evaluated as cohesive units with ranks, relevance, and pairing analysis.")
        if os.path.exists(RANKED_PROMOS_FILE):
            with open(RANKED_PROMOS_FILE, "r", encoding="utf-8") as f:
                ranked_json_str = f.read()
            st.download_button(
                label="📥 Download ranked_promos.json",
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
            st.info("`ranked_promos.json` will be generated after running the pipeline.")

    with col_deliv2:
        st.markdown("### 2. `extracted_promos.json`")
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
            st.info("`extracted_promos.json` will be generated after running the pipeline.")

# TAB 4: XML SCREENPLAY CORPUS INSPECTOR
with tab_xml:
    st.subheader("📑 Screenplay Corpus Inspector")
    st.markdown("The complete series screenplay compiled into a unified XML document, enabling macro narrative analysis across all episodes.")

    corpus_text = st.session_state.get("xml_corpus", "")
    if not corpus_text and selected_episodes:
        corpus_text = build_xml_tagged_corpus(selected_episodes)

    if corpus_text:
        st.caption(f"Corpus Length: {len(corpus_text):,} characters (~{len(corpus_text)//4:,} tokens)")
        st.text_area("XML Corpus Preview", value=corpus_text[:12000] + ("\n\n... [truncated for preview] ..." if len(corpus_text) > 12000 else ""), height=450)
    else:
        st.info("No XML corpus available yet.")
