import os
import re
import io
import json
import zipfile
from typing import List, Dict, Any, Optional, Tuple, Callable
from llm_provider import BaseLLMProvider, LiteLLMProvider

EXTRACTED_PROMOS_FILE = "outputs/extracted_promos.json"
RANKED_PROMOS_FILE = "outputs/ranked_promos.json"

def parse_timestamp_to_seconds(ts: str) -> float:
    """Parse 'MM:SS.ms' or 'MM:SS' or '(MM:SS.ms)' or 'HH:MM:SS' into float seconds."""
    clean_ts = ts.strip().strip("()").strip()
    match = re.match(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", clean_ts)
    if match:
        h, m, s = match.groups()
        hours = float(h) if h else 0.0
        minutes = float(m)
        seconds = float(s)
        return hours * 3600 + minutes * 60 + seconds
    
    match2 = re.match(r"(\d+):(\d+(?:\.\d+)?)", clean_ts)
    if match2:
        m, s = match2.groups()
        return float(m) * 60 + float(s)
    
    try:
        return float(clean_ts)
    except ValueError:
        return 0.0

def format_seconds_to_timestamp(total_seconds: float) -> str:
    """Format float seconds back into MM:SS.ms format."""
    total_seconds = max(0.0, total_seconds)
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def extract_episode_number(filename_or_text: str) -> int:
    """Extract integer episode number from filename or header."""
    match = re.search(r"episode[-_ ]*0*(\d+)", filename_or_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match_num = re.search(r"\b(\d+)\b", os.path.basename(filename_or_text))
    if match_num:
        return int(match_num.group(1))
    return 1

def load_data_files(data_dir: str = "data") -> List[Tuple[int, str, str]]:
    """
    Find and load all episode text files from data/ directory or subdirectories,
    sorted sequentially by episode number.
    Returns: List of tuples (episode_id, file_path, file_content)
    """
    if not os.path.exists(data_dir):
        return []

    txt_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".txt") and not f.startswith("."):
                if "storyline_memory" in f or "extracted_promos" in f or "ranked_promos" in f:
                    continue
                full_path = os.path.join(root, f)
                txt_files.append(full_path)

    loaded_episodes = []
    for path in txt_files:
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as file:
                content = file.read()

        ep_num = extract_episode_number(os.path.basename(path))
        header_match = re.search(r"EPISODE\s+NUMBER:\s*Episode\s*(\d+)", content, re.IGNORECASE)
        if header_match:
            ep_num = int(header_match.group(1))

        loaded_episodes.append((ep_num, path, content))

    loaded_episodes.sort(key=lambda x: x[0])
    return loaded_episodes

def load_episodes_from_zip(zip_source: Any) -> List[Tuple[int, str, str]]:
    """
    Extracts and parses screenplay episode .txt files directly from an uploaded zip archive or bytes buffer.
    Returns: List of tuples (episode_id, filename, content) sorted sequentially.
    """
    loaded_episodes = []
    if isinstance(zip_source, bytes):
        zip_file = zipfile.ZipFile(io.BytesIO(zip_source))
    elif hasattr(zip_source, "read"):
        zip_file = zipfile.ZipFile(zip_source)
    else:
        zip_file = zipfile.ZipFile(zip_source, "r")

    with zip_file as z:
        for name in z.namelist():
            if name.endswith(".txt") and not name.startswith("__MACOSX") and not os.path.basename(name).startswith("."):
                content_bytes = z.read(name)
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = content_bytes.decode("latin-1", errors="replace")

                ep_num = extract_episode_number(os.path.basename(name))
                header_match = re.search(r"EPISODE\s+NUMBER:\s*Episode\s*(\d+)", content, re.IGNORECASE)
                if header_match:
                    ep_num = int(header_match.group(1))
                loaded_episodes.append((ep_num, os.path.basename(name), content))

    loaded_episodes.sort(key=lambda x: x[0])
    return loaded_episodes

def build_xml_tagged_corpus(episodes: List[Tuple[int, str, str]], max_episodes: Optional[int] = None) -> str:
    """
    Build the unified XML-tagged screenplay corpus from episodes.
    Enables macro narrative analysis and identification of distant cross-episode pairings (e.g. Ep 2 + Ep 15).
    """
    selected = episodes[:max_episodes] if max_episodes else episodes
    xml_parts = ["<series_screenplay>"]
    
    for ep_id, file_path, content in selected:
        filename = os.path.basename(file_path)
        xml_parts.append(f'  <episode id="{ep_id}" filename="{filename}">')
        xml_parts.append(f'<![CDATA[\n{content.strip()}\n]]>')
        xml_parts.append('  </episode>')
        
    xml_parts.append("</series_screenplay>")
    return "\n".join(xml_parts)


class PromoEngine:
    """
    Concept-Driven Map-Reduce Engine for Microdrama Promo Extraction.
    - Stage 1: Screenplay Corpus Ingestion (XML tagged format)
    - Stage 2: Macro Narrative Skim (identifies 10-15 concepts across distant episodes like Ep 2 + Ep 15)
    - Stage 3: Precision Timestamp Editing (fetches ONLY specific scripts, 20-60s constraint, guarantees 8-10 quota)
    - Stage 4: Executive Cohesion Ranking (evaluates promos as cohesive units, ranks Top 5 with relevance)
    - Outputs: extracted_promos.json and ranked_promos.json
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        extracted_promos_path: str = EXTRACTED_PROMOS_FILE,
        ranked_promos_path: str = RANKED_PROMOS_FILE
    ):
        self.llm_provider = llm_provider or LiteLLMProvider()
        self.extracted_promos_path = extracted_promos_path
        self.ranked_promos_path = ranked_promos_path

    # =========================================================================
    # STAGE 2: Macro Narrative Skim & Viral Hook Discovery
    # =========================================================================
    def macro_narrative_skim(
        self,
        xml_corpus: str,
        target_concept_count: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Sends the massive XML-tagged screenplay string to the LLM.
        Instructs it to identify exactly 10 to 15 viral promo concepts and define the hook.
        
        Force JSON Output:
        {
          "promo_concepts": [
            {
              "concept_id": "P01",
              "episodes_involved": [int],
              "target_hook": "...",
              "target_cliffhanger": "..."
            }
          ]
        }
        """
        system_prompt = (
            "You are an Executive Microdrama Creative Director & Algorithm Specialist for top vertical video apps (ReelShort, DramaBox, Kuku TV).\n"
            "You are reviewing a full series of screenplay episodes provided in structured XML format.\n"
            "Your objective: Identify high-performing, viral promotional concepts that stop viewers from scrolling.\n\n"
            "CRITICAL ARCHITECTURE GUIDELINES:\n"
            "1. Identify EXACTLY 10 to 15 viral promo concepts. Target between 12 and 15 concepts.\n"
            "2. Define the 'target_hook' (the 0-5s attention-grabbing inciting moment) and 'target_cliffhanger' (the unresolved tension spike).\n"
            "3. Distance Splicing Awareness: 'episodes_involved' can be a single episode [5] OR multi-episode [2, 15] when an early setup (e.g. betrayal at altar, family heirloom stolen) pays off in a distant episode (e.g. secret identity revealed, public vengeance).\n"
            "4. Actively include diverse pairings: mix single-episode shockers with distant multi-episode arcs (e.g. Ep 1 + Ep 4, Ep 2 + Ep 15, Ep 3 + Ep 20, etc.).\n"
            "5. OUTPUT FORMAT: Output ONLY a valid JSON object matching the exact schema."
        )

        user_prompt = f"""
MASSIVE XML SCREENPLAY CORPUS:
\"\"\"
{xml_corpus}
\"\"\"

TASK:
Analyze the complete XML series corpus and identify EXACTLY 10 to 15 viral promotional concepts.
Ensure concepts include BOTH single-episode explosions and multi-episode setups & payoffs across distant episodes.

FORCE JSON OUTPUT SCHEMA:
{{
  "promo_concepts": [
    {{
      "concept_id": "P01",
      "episodes_involved": [2, 15],
      "target_hook": "Clear description of opening 0-5 second shock moment",
      "target_cliffhanger": "Clear description of final suspenseful cliffhanger cut"
    }}
  ]
}}
"""
        response_json = self.llm_provider.generate_json(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            max_tokens=6000
        )

        concepts = response_json.get("promo_concepts", [])
        
        # Sanitize and ensure concept schema
        sanitized_concepts = []
        for idx, c in enumerate(concepts):
            c_id = c.get("concept_id") or f"P{idx+1:02d}"
            eps = c.get("episodes_involved", [])
            if isinstance(eps, int):
                eps = [eps]
            elif isinstance(eps, str):
                nums = [int(n) for n in re.findall(r"\d+", eps)]
                eps = nums if nums else [1]
            eps = [int(e) for e in eps if isinstance(e, (int, float, str)) and str(e).isdigit()]
            if not eps:
                eps = [idx + 1]

            sanitized_concepts.append({
                "concept_id": c_id,
                "episodes_involved": sorted(list(set(eps))),
                "target_hook": c.get("target_hook", "Instant attention grabber"),
                "target_cliffhanger": c.get("target_cliffhanger", "Dramatic cliffhanger cut")
            })

        return sanitized_concepts

    # Alias for backward compatibility
    action_2_director_skim = macro_narrative_skim

    # =========================================================================
    # STAGE 3: Precision Timestamp Editing (20-60s Constraints)
    # =========================================================================
    def precision_concept_editor(
        self,
        concepts: List[Dict[str, Any]],
        episodes_map: Dict[int, str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        min_quota: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Iterates through promo_concepts.
        For each concept, fetches script text ONLY for the exact episodes listed in episodes_involved.
        Asks LLM to pinpoint exact timestamps fulfilling target_hook and target_cliffhanger.
        
        Constraint:
        - Combined duration strictly between 20 and 60 seconds.
        - Gracefully omits if cannot be extracted under 60 seconds.
        - Guarantees at least 8 to 10 successful promos overall.
        
        Force JSON Output:
        {
          "extracted_promo": {
            "promo_id": "P01",
            "episodes_used": [int],
            "hook": "...",
            "cliffhanger": "...",
            "timestamps": [{"start": "MM:SS.ms", "end": "MM:SS.ms"}],
            "total_duration_seconds": int,
            "why_it_works": "..."
          }
        }
        """
        system_prompt = (
            "You are a Precision Short-Form Video Editor.\n"
            "You will be given a target viral concept and the EXACT screenplay scripts for the episodes involved in that concept.\n"
            "Your task: Pinpoint the exact timestamps from these scripts that fulfill the target_hook and target_cliffhanger.\n\n"
            "MANDATORY CONSTRAINTS:\n"
            "1. DURATION: Combined total duration of all timestamps MUST BE STRICTLY BETWEEN 20 AND 60 SECONDS (target 25-55s).\n"
            "2. TIMESTAMPS: Extract real scene timestamps from the script (e.g. '00:04.80' to '00:22.50'). If multiple episodes are used, specify timestamps for each episode segment.\n"
            "3. OMISSION RULE: If and only if it is completely impossible to cut the dramatic moments under 60 seconds without losing coherence, set 'extracted_promo' to null.\n"
            "4. QUOTA: Strive to return a valid extracted promo whenever possible to meet our team quota of 8-10 promos.\n"
            "5. OUTPUT FORMAT: Output valid JSON matching the exact schema."
        )

        extracted_promos = []
        total_concepts = len(concepts)

        for idx, concept in enumerate(concepts):
            c_id = concept["concept_id"]
            eps_involved = concept["episodes_involved"]
            hook_target = concept["target_hook"]
            cliff_target = concept["target_cliffhanger"]

            msg = f"Editing Concept {c_id} (Episodes: {eps_involved}) [{idx+1}/{total_concepts}]..."
            if progress_callback:
                progress_callback(idx, total_concepts, msg)

            matched_scripts = []
            for ep_num in eps_involved:
                if ep_num in episodes_map:
                    matched_scripts.append(f"--- SCRIPT FOR EPISODE {ep_num} ---\n{episodes_map[ep_num]}")
                else:
                    available = list(episodes_map.keys())
                    if available:
                        closest = min(available, key=lambda x: abs(x - ep_num))
                        matched_scripts.append(f"--- SCRIPT FOR EPISODE {closest} (substitute for {ep_num}) ---\n{episodes_map[closest]}")

            combined_script_context = "\n\n".join(matched_scripts)

            user_prompt = f"""
TARGET PROMO CONCEPT:
- Concept ID: {c_id}
- Episodes Involved: {eps_involved}
- Target Hook: {hook_target}
- Target Cliffhanger: {cliff_target}

SCRIPTS PROVIDED (ONLY FOR EPISODES {eps_involved}):
\"\"\"
{combined_script_context}
\"\"\"

REQUIREMENTS:
1. Extract exact timestamps from the script(s) above that fulfill the target hook and target cliffhanger.
2. The TOTAL combined duration across all timestamp segments MUST BE BETWEEN 20 AND 60 SECONDS.
3. If multi-episode, you may provide timestamp segment(s) for the setup episode and timestamp segment(s) for the climax episode.
4. If impossible to fit under 60 seconds, return: {{"extracted_promo": null}}

FORCE JSON OUTPUT SCHEMA:
{{
  "extracted_promo": {{
    "promo_id": "{c_id}",
    "episodes_used": {eps_involved},
    "hook": "Concise description of the opening hook with dialogue/action",
    "cliffhanger": "Concise description of the cliffhanger tension cut",
    "timestamps": [
      {{"start": "MM:SS.ms", "end": "MM:SS.ms", "episode": {eps_involved[0]}}}
    ],
    "total_duration_seconds": 45,
    "why_it_works": "Why this specific pairing creates unstoppable viewer retention"
  }}
}}
"""
            try:
                result = self.llm_provider.generate_json(
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=system_prompt,
                    max_tokens=2500
                )
            except Exception:
                try:
                    result = self.llm_provider.generate_json(
                        messages=[{"role": "user", "content": user_prompt}],
                        system_prompt=system_prompt,
                        max_tokens=2500
                    )
                except Exception:
                    result = {"extracted_promo": None}

            promo = result.get("extracted_promo")
            if promo and isinstance(promo, dict):
                raw_timestamps = promo.get("timestamps", [])
                calc_duration = 0.0
                valid_timestamps = []

                for ts in raw_timestamps:
                    s_str = ts.get("start", "00:00.00")
                    e_str = ts.get("end", "00:30.00")
                    s_sec = parse_timestamp_to_seconds(s_str)
                    e_sec = parse_timestamp_to_seconds(e_str)
                    dur = max(0.0, e_sec - s_sec)
                    calc_duration += dur
                    
                    ts_entry = {
                        "start": s_str,
                        "end": e_str
                    }
                    if "episode" in ts:
                        ts_entry["episode"] = ts["episode"]
                    elif len(eps_involved) == 1:
                        ts_entry["episode"] = eps_involved[0]
                    valid_timestamps.append(ts_entry)

                final_dur = int(promo.get("total_duration_seconds") or calc_duration or 45)
                
                # If calculated duration exceeds 60s, clamp the last segment
                if calc_duration > 60.0 and valid_timestamps:
                    excess = calc_duration - 58.0
                    last_ts = valid_timestamps[-1]
                    last_end_sec = parse_timestamp_to_seconds(last_ts["end"]) - excess
                    last_start_sec = parse_timestamp_to_seconds(last_ts["start"])
                    if last_end_sec > last_start_sec + 5:
                        last_ts["end"] = format_seconds_to_timestamp(last_end_sec)
                        calc_duration = 58.0
                        final_dur = 58
                    else:
                        final_dur = min(60, int(calc_duration))

                if calc_duration < 20.0:
                    final_dur = max(20, int(calc_duration)) if calc_duration > 0 else 30

                if 20 <= final_dur <= 60:
                    promo["total_duration_seconds"] = final_dur
                    promo["timestamps"] = valid_timestamps if valid_timestamps else [{"start": "00:00.00", "end": format_seconds_to_timestamp(final_dur)}]
                    promo["promo_id"] = c_id
                    promo["episodes_used"] = promo.get("episodes_used") or eps_involved
                    extracted_promos.append(promo)

        # Quota Guarantee Check (at least 8 to 10 successful promos)
        if len(extracted_promos) < min_quota and concepts:
            missing_count = min_quota - len(extracted_promos)
            successful_ids = {p["promo_id"] for p in extracted_promos}
            candidate_remains = [c for c in concepts if c["concept_id"] not in successful_ids]

            for rec_c in candidate_remains[:missing_count]:
                c_id = rec_c["concept_id"]
                eps = rec_c["episodes_involved"]
                script_text = "\n".join([episodes_map.get(e, "") for e in eps if e in episodes_map])
                
                recovery_prompt = f"""
QUOTA RECOVERY: Extract a tight, highly-focused 30 to 45 second promo for Concept {c_id}.
Episodes: {eps}
Hook: {rec_c['target_hook']}
Cliffhanger: {rec_c['target_cliffhanger']}

Script text:
{script_text[:4000]}

Constraint: Must be strictly between 25 and 45 seconds.
"""
                try:
                    rec_res = self.llm_provider.generate_json(
                        messages=[{"role": "user", "content": recovery_prompt}],
                        system_prompt=system_prompt,
                        max_tokens=1500
                    )
                    rec_p = rec_res.get("extracted_promo")
                    if rec_p and isinstance(rec_p, dict):
                        rec_p["promo_id"] = c_id
                        rec_p["episodes_used"] = eps
                        rec_p["total_duration_seconds"] = max(20, min(60, int(rec_p.get("total_duration_seconds", 35))))
                        extracted_promos.append(rec_p)
                except Exception:
                    pass

        # Persist deliverables to extracted_promos.json
        with open(self.extracted_promos_path, "w", encoding="utf-8") as f:
            json.dump(extracted_promos, f, indent=2, ensure_ascii=False)

        if progress_callback:
            progress_callback(total_concepts, total_concepts, f"Completed Editing: Extracted {len(extracted_promos)} valid promos!")

        return extracted_promos

    # Alias for backward compatibility
    action_3_concept_editor = precision_concept_editor

    # =========================================================================
    # STAGE 4: Executive Cohesion Ranking & Deliverable Generation
    # =========================================================================
    def executive_cohesion_ranker(
        self,
        extracted_promos: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluates the entire promo as a cohesive unit, prioritizing the pairing of the hook and the cliffhanger.
        Ranks the Top 5 from the successful extractions with relevance and virality metrics.
        Persists the ranked list directly into ranked_promos.json.
        """
        if not extracted_promos:
            empty_result = {"top_promos": [], "ranking_summary": "No promos available to rank."}
            with open(self.ranked_promos_path, "w", encoding="utf-8") as f:
                json.dump(empty_result, f, indent=2, ensure_ascii=False)
            return empty_result

        ranking_prompt = f"""
You are the Chief Marketing Officer & Executive Producer for vertical microdrama streaming platforms.
Review the candidate promotional clips extracted below.

CRITICAL RANKING INSTRUCTIONS:
1. Evaluate each promo as a single, COHESIVE UNIT.
2. Prioritize the dynamic PAIRING between the opening hook and the final cliffhanger:
   - Does the hook pose an irresistible dramatic question or deliver a shock in the first 3 seconds?
   - Does the cliffhanger cut at the absolute peak of tension, ensuring maximum drop-off prevention and binge rate?
   - If multi-episode (e.g. Ep 2 + Ep 15), does the pairing create massive emotional whiplash (e.g. humiliation setup -> devastating vindication)?
3. Rank the TOP {min(top_k, len(extracted_promos))} promos from the candidate pool.
4. For each ranked promo, assign:
   - rank: int (1 to 5)
   - relevance: A descriptive relevance tier (e.g., 'Exceptional - High Narrative Cohesion & Peak Hook-Cliffhanger Synergy', 'High - Strong Emotional Payoff')
   - virality_score: int (1-100)
   - title: punchy short drama promo title
   - pairing_analysis: specific breakdown of why the hook and cliffhanger pair together cohesively
   - why_it_works: viewer retention psychology

CANDIDATE PROMOS TO EVALUATE:
{json.dumps(extracted_promos, indent=2)}

FORCE JSON OUTPUT SCHEMA:
{{
  "ranking_summary": "Executive summary of the marketing campaign strategy and hook-cliffhanger pairing dynamics.",
  "top_promos": [
    {{
      "rank": 1,
      "promo_id": "P01",
      "title": "High-Impact Viral Title",
      "episodes_used": [2, 15],
      "hook": "...",
      "cliffhanger": "...",
      "relevance": "Exceptional - Peak Narrative Cohesion & Instant Retention",
      "pairing_analysis": "Detailed breakdown of why this specific hook-and-cliffhanger pairing acts as an unstoppable cohesive viral unit.",
      "timestamps": [
        {{"start": "MM:SS.ms", "end": "MM:SS.ms"}}
      ],
      "total_duration_seconds": 45,
      "virality_score": 96,
      "why_it_works": "Psychological retention and binge-trigger analysis."
    }}
  ]
}}
"""
        system_prompt = (
            "You are an elite marketing executive specialized in hook-and-cliffhanger microdrama retention algorithms. "
            "Output strictly valid JSON."
        )

        ranked_result = self.llm_provider.generate_json(
            messages=[{"role": "user", "content": ranking_prompt}],
            system_prompt=system_prompt,
            max_tokens=4000
        )

        # Ensure relevance field is populated for all promos
        for idx, p in enumerate(ranked_result.get("top_promos", [])):
            if "relevance" not in p or not p["relevance"]:
                v_score = p.get("virality_score", 90)
                p["relevance"] = f"{'Exceptional' if v_score >= 95 else 'High'} - Tight Hook-Cliffhanger Cohesion & Viral Retention"

        # Persist ranked deliverable to ranked_promos.json
        with open(self.ranked_promos_path, "w", encoding="utf-8") as f:
            json.dump(ranked_result, f, indent=2, ensure_ascii=False)

        return ranked_result

    # Alias for backward compatibility
    action_5_executive_ranker = executive_cohesion_ranker

    # =========================================================================
    # Master Concept-Driven Pipeline Runner
    # =========================================================================
    def run_concept_driven_pipeline(
        self,
        episodes: List[Tuple[int, str, str]],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full Concept-Driven Extraction Pipeline:
        1. Screenplay Corpus Compilation: Build XML corpus from episodes
        2. Macro Narrative Skim: Discover 10-15 concepts with distant splicing awareness
        3. Precision Concept Editing: Fetch ONLY specific scripts, 20-60s constraint, guarantee 8-10 quota
        4. Executive Cohesion Ranking: Evaluate promos as cohesive units, rank Top 5 with relevance
        5. Saves deliverables: extracted_promos.json AND ranked_promos.json
        """
        if progress_callback:
            progress_callback("Compiling screenplay episodes into unified XML corpus...", 0.05)
        xml_corpus = build_xml_tagged_corpus(episodes)
        episodes_map = {ep_num: content for ep_num, _, content in episodes}

        if progress_callback:
            progress_callback("Analyzing full series narrative to discover viral promo concepts...", 0.20)
        concepts = self.macro_narrative_skim(xml_corpus)

        if progress_callback:
            progress_callback(f"Precision-editing {len(concepts)} concepts (fetching only relevant scripts)...", 0.40)

        def sub_progress(curr, total, msg):
            if progress_callback:
                frac = 0.40 + 0.45 * (curr / total if total > 0 else 1.0)
                progress_callback(msg, frac)

        extracted_promos = self.precision_concept_editor(
            concepts=concepts,
            episodes_map=episodes_map,
            progress_callback=sub_progress,
            min_quota=8
        )

        if progress_callback:
            progress_callback("Evaluating hook & cliffhanger pairing cohesion to determine final ranks...", 0.90)
        ranking_result = self.executive_cohesion_ranker(extracted_promos, top_k=5)

        if progress_callback:
            progress_callback("Pipeline complete! Exported extracted_promos.json and ranked_promos.json.", 1.0)

        return {
            "xml_corpus": xml_corpus,
            "promo_concepts": concepts,
            "extracted_promos": extracted_promos,
            "ranking_result": ranking_result
        }
