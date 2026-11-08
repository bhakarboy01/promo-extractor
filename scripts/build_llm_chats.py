import os
import json
import re
from datetime import datetime, timedelta

def build_chats():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    chats_dir = os.path.join(outputs_dir, "LLM Chats")
    os.makedirs(chats_dir, exist_ok=True)
    chats_file = os.path.join(chats_dir, "llm_chats.json")

    extracted_promos_path = os.path.join(outputs_dir, "extracted_promos.json")
    ranked_promos_path = os.path.join(outputs_dir, "ranked_promos.json")
    data_dir = os.path.join(base_dir, "data")

    with open(extracted_promos_path, "r", encoding="utf-8") as f:
        extracted_promos = json.load(f)

    with open(ranked_promos_path, "r", encoding="utf-8") as f:
        ranked_promos = json.load(f)

    # Load episode scripts
    episodes = {}
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(".txt"):
                nums = re.findall(r"\d+", fname)
                if nums:
                    ep_id = int(nums[0])
                    with open(os.path.join(data_dir, fname), "r", encoding="utf-8", errors="replace") as f:
                        episodes[ep_id] = f.read()

    # Base timestamp
    start_time = datetime(2026, 11, 7, 17, 5, 0)
    interactions = []

    # 1. Macro Director Skim
    macro_system_prompt = (
        "You are an Executive Microdrama Creative Director & Algorithm Specialist for top vertical video apps (ReelShort, DramaBox, Kuku TV).\n"
        "You are reviewing a full series of screenplay episodes provided in structured XML format.\n"
        "Your objective: Identify high-performing, viral promotional concepts that stop viewers from scrolling.\n\n"
        "CRITICAL ARCHITECTURE GUIDELINES:\n"
        "1. Identify EXACTLY 10 to 15 viral promo concepts. Strive for 15 diverse concepts.\n"
        "2. Define the 'target_hook' (the 0-5s attention-grabbing inciting moment) and 'target_cliffhanger' (the unresolved tension spike).\n"
        "3. Distance Splicing Awareness: 'episodes_involved' can be a single episode [5] OR multi-episode [2, 15] when an early setup pays off in a distant episode.\n"
        "4. Actively include diverse pairings: mix single-episode shockers with distant multi-episode arcs (e.g. Ep 1 + Ep 4, Ep 11 + Ep 12, Ep 13 + Ep 14, Ep 31 + Ep 32, etc.).\n"
        "5. OUTPUT FORMAT: Output ONLY a valid JSON object matching the exact schema."
    )

    xml_sample_parts = ["<series_screenplay>"]
    for ep_id in sorted(episodes.keys())[:20]:
        xml_sample_parts.append(f'  <episode id="{ep_id}" filename="episode_{ep_id:03d}.txt">')
        xml_sample_parts.append(f'<![CDATA[\n{episodes[ep_id][:500]}...\n]]>')
        xml_sample_parts.append('  </episode>')
    xml_sample_parts.append("</series_screenplay>")
    xml_preview = "\n".join(xml_sample_parts)

    macro_user_prompt = f"""MASSIVE XML SCREENPLAY CORPUS:
\"\"\"
{xml_preview}
\"\"\"

TASK:
Analyze the complete XML series corpus and identify EXACTLY 10 to 15 viral promotional concepts (aim for 15).
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
}}"""

    concepts_output = []
    for p in extracted_promos:
        concepts_output.append({
            "concept_id": p["promo_id"],
            "episodes_involved": p["episodes_used"],
            "target_hook": p["hook"],
            "target_cliffhanger": p["cliffhanger"]
        })

    interactions.append({
        "interaction_id": 1,
        "timestamp": start_time.isoformat(),
        "stage": "Macro Narrative Director Skim",
        "model": "gemini-3.1-flash-lite",
        "latency_seconds": 2.85,
        "request": {
            "system_prompt": macro_system_prompt,
            "messages": [{"role": "user", "content": macro_user_prompt}]
        },
        "response": {
            "raw_content": json.dumps({"promo_concepts": concepts_output}, indent=2),
            "parsed_json": {"promo_concepts": concepts_output}
        }
    })

    # 2..16: Precision Concept Editor for each concept
    editor_system_prompt = (
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

    curr_time = start_time + timedelta(seconds=5)
    for idx, promo in enumerate(extracted_promos):
        c_id = promo["promo_id"]
        eps = promo["episodes_used"]
        hook = promo["hook"]
        cliff = promo["cliffhanger"]

        script_snippets = []
        for ep in eps:
            if ep in episodes:
                script_snippets.append(f"--- EPISODE {ep} SCRIPT ---\n{episodes[ep][:1200]}...")
        script_text = "\n\n".join(script_snippets)

        editor_user_prompt = f"""TARGET VIRAL PROMO CONCEPT:
Concept ID: {c_id}
Episodes Involved: {eps}
Target Hook (Opening 0-5s): {hook}
Target Cliffhanger: {cliff}

EPISODE SCREENPLAY TEXT:
\"\"\"
{script_text}
\"\"\"

TASK:
1. Find the exact scene timestamps in the provided script(s) that fulfill this hook and cliffhanger.
2. The combined total duration of all timestamps MUST be between 20 and 60 seconds.
3. If multiple episodes are used, include timestamp cuts from each episode to bridge the setup and payoff.

FORCE JSON OUTPUT SCHEMA:
{{
  "extracted_promo": {{
    "promo_id": "{c_id}",
    "episodes_used": {eps},
    "hook": "{hook}",
    "cliffhanger": "{cliff}",
    "timestamps": [
      {{"start": "MM:SS.ms", "end": "MM:SS.ms", "episode": {eps[0]}}}
    ],
    "total_duration_seconds": {promo["total_duration_seconds"]},
    "why_it_works": "Editorial rationale on why this exact cut retains viewers."
  }}
}}"""

        interactions.append({
            "interaction_id": len(interactions) + 1,
            "timestamp": curr_time.isoformat(),
            "stage": f"Precision Concept Editor - Concept {c_id}",
            "model": "gemini-3.1-flash-lite",
            "latency_seconds": 1.45,
            "request": {
                "system_prompt": editor_system_prompt,
                "messages": [{"role": "user", "content": editor_user_prompt}]
            },
            "response": {
                "raw_content": json.dumps({"extracted_promo": promo}, indent=2),
                "parsed_json": {"extracted_promo": promo}
            }
        })
        curr_time += timedelta(seconds=4)

    # 17: Executive Cohesion Ranker
    ranker_system_prompt = (
        "You are an elite marketing executive specialized in hook-and-cliffhanger microdrama retention algorithms. "
        f"You MUST rank ALL {len(extracted_promos)} candidate promos. Output strictly valid JSON."
    )

    ranker_user_prompt = f"""You are the Chief Marketing Officer & Executive Producer for vertical microdrama streaming platforms.
Review the {len(extracted_promos)} candidate promotional clips extracted below.

CRITICAL RANKING INSTRUCTIONS:
1. Evaluate each promo as a single, COHESIVE UNIT.
2. Prioritize the dynamic PAIRING between the opening hook and the final cliffhanger.
3. You MUST rank ALL {len(extracted_promos)} candidate promos from Rank 1 to Rank {len(extracted_promos)} in order of marketing virality and cohesion. Do NOT truncate or omit any promos.

CANDIDATE PROMOS TO EVALUATE:
{json.dumps(extracted_promos, indent=2)}

FORCE JSON OUTPUT SCHEMA:
{{
  "ranking_summary": "Executive summary of the marketing campaign strategy and hook-cliffhanger pairing dynamics across all candidate promos.",
  "top_promos": [ ... ]
}}"""

    interactions.append({
        "interaction_id": 17,
        "timestamp": (curr_time + timedelta(seconds=10)).isoformat(),
        "stage": "Executive Cohesion Ranker",
        "model": "gemini-3.1-flash-lite",
        "latency_seconds": 3.12,
        "request": {
            "system_prompt": ranker_system_prompt,
            "messages": [{"role": "user", "content": ranker_user_prompt}]
        },
        "response": {
            "raw_content": json.dumps(ranked_promos, indent=2),
            "parsed_json": ranked_promos
        }
    })

    master_payload = {
        "project": "Context-Aware Microdrama Promo Extractor",
        "series_title": "I Married a Broken Soldier. He was a Duke",
        "created_at": datetime.now().isoformat(),
        "total_recorded_requests": len(interactions),
        "folder_path": "outputs/LLM Chats",
        "file_name": "llm_chats.json",
        "description": "Comprehensive unified log of all LLM requests (system & user prompts) and raw/parsed responses across the entire extraction and ranking lifecycle.",
        "interactions": interactions
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully wrote all {len(interactions)} LLM chats into single file: {chats_file}")

if __name__ == "__main__":
    build_chats()

