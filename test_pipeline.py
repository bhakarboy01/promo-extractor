import os
import io
import json
import zipfile
from promo_engine import (
    parse_timestamp_to_seconds,
    format_seconds_to_timestamp,
    extract_episode_number,
    load_data_files,
    load_episodes_from_zip,
    build_xml_tagged_corpus,
    PromoEngine
)

def test_timestamp_utils():
    assert parse_timestamp_to_seconds("00:00.00") == 0.0
    assert parse_timestamp_to_seconds("01:30.00") == 90.0
    assert parse_timestamp_to_seconds("02:18.50") == 138.5
    assert format_seconds_to_timestamp(90.0) == "01:30.00"
    print("✓ Timestamp tests passed!")

def test_zip_loader():
    # Create in-memory zip with 2 sample episodes
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("episode-001.txt", "EPISODE NUMBER: Episode 1\nScene 1\nStart: 00:00.00 End: 00:30.00")
        z.writestr("episode-002.txt", "EPISODE NUMBER: Episode 2\nScene 1\nStart: 00:00.00 End: 00:45.00")
    buf.seek(0)

    episodes = load_episodes_from_zip(buf.getvalue())
    assert len(episodes) == 2, f"Expected 2 episodes from zip, got {len(episodes)}"
    assert episodes[0][0] == 1
    assert episodes[1][0] == 2
    print(f"✓ ZIP loader test passed! Successfully unpacked {len(episodes)} episodes.")

def test_xml_corpus_builder():
    episodes = load_data_files("data")[:5]
    xml_corpus = build_xml_tagged_corpus(episodes)
    assert "<series_screenplay>" in xml_corpus
    assert '<episode id="1"' in xml_corpus
    assert '<episode id="5"' in xml_corpus
    assert "</series_screenplay>" in xml_corpus
    print(f"✓ XML corpus builder test passed! Generated {len(xml_corpus)} characters of XML for 5 episodes.")

def test_deliverable_files():
    # Check outputs subfolder deliverables
    outputs_extracted = os.path.join("outputs", "extracted_promos.json")
    outputs_ranked = os.path.join("outputs", "ranked_promos.json")
    extras_dir = os.path.join("outputs", "extras")
    outputs_chats = os.path.join(extras_dir, "llm_chats.json")
    outputs_xml = os.path.join(extras_dir, "series_screenplay_corpus.xml")
    outputs_prompts = os.path.join(extras_dir, "prompt_templates.md")
    outputs_tour = os.path.join(extras_dir, "presentation_tour_script.txt")
    outputs_mermaid = os.path.join(extras_dir, "architecture_flowchart.mmd")
    outputs_sol_md = os.path.join(extras_dir, "solution_doc.md")
    outputs_sol_tex = os.path.join(extras_dir, "solution_doc.tex")

    assert os.path.exists(outputs_extracted), f"{outputs_extracted} must exist"
    assert os.path.exists(outputs_ranked), f"{outputs_ranked} must exist"
    assert os.path.exists(outputs_chats), f"{outputs_chats} must exist"
    assert os.path.exists(outputs_xml), f"{outputs_xml} must exist"
    assert os.path.exists(outputs_prompts), f"{outputs_prompts} must exist"
    assert os.path.exists(outputs_tour), f"{outputs_tour} must exist"
    assert os.path.exists(outputs_mermaid), f"{outputs_mermaid} must exist"
    assert os.path.exists(outputs_sol_md), f"{outputs_sol_md} must exist"
    assert os.path.exists(outputs_sol_tex), f"{outputs_sol_tex} must exist"
    
    with open(outputs_extracted, "r", encoding="utf-8") as f:
        ext = json.load(f)
    with open(outputs_ranked, "r", encoding="utf-8") as f:
        rnk = json.load(f)
    with open(outputs_chats, "r", encoding="utf-8") as f:
        chats = json.load(f)
        
    assert len(ext) >= 8, f"Must meet minimum 8-10 promo quota, found {len(ext)}"
    assert "top_promos" in rnk, "ranked_promos.json must contain top_promos"
    assert len(rnk["top_promos"]) == 15, f"All 15 promos must be ranked, found {len(rnk['top_promos'])}"
    assert "interactions" in chats, "llm_chats.json must contain interactions"
    assert len(chats["interactions"]) >= 15, f"Expected at least 15 interactions, found {len(chats['interactions'])}"
    print(f"✓ Deliverables verified in outputs/ and outputs/extras/! extracted_promos.json ({len(ext)} promos), ranked_promos.json ({len(rnk['top_promos'])} ranked promos), llm_chats.json ({len(chats['interactions'])} chats), XML corpus ({os.path.getsize(outputs_xml):,} bytes).")

def test_dynamic_model_routing():
    from llm_provider import LiteLLMProvider
    provider = LiteLLMProvider()

    # 1. Verify default routing
    assert provider.DEFAULT_ROUTING["Macro Narrative Director Skim"] == "gemini/gemini-1.5-flash"
    assert provider.DEFAULT_ROUTING["Precision Concept Editor"] == "gemini/gemini-1.5-pro"
    assert provider.DEFAULT_ROUTING["Executive Cohesion Ranker"] == "gemini/gemini-1.5-pro"
    assert len(provider.custom_routing) == 0

    # 2. Test custom stage route
    provider.set_custom_route("Precision Concept Editor", "gpt-4o")
    assert provider.custom_routing["Precision Concept Editor"] == "gpt-4o"
    provider.set_custom_route("Precision Concept Editor", None)
    assert "Precision Concept Editor" not in provider.custom_routing

    # 3. Test global override
    provider.set_global_override("claude-3-5-sonnet-20240620")
    for s in provider.DEFAULT_ROUTING.keys():
        assert provider.custom_routing[s] == "claude-3-5-sonnet-20240620"
    
    # Clear global override
    provider.set_global_override("")
    assert len(provider.custom_routing) == 0

    # 4. Test API key injection per model prefix
    provider._inject_api_key_for_model("gpt-4o", "test-openai-key")
    assert os.environ.get("OPENAI_API_KEY") == "test-openai-key"

    provider._inject_api_key_for_model("claude-3-5-sonnet", "test-anthropic-key")
    assert os.environ.get("ANTHROPIC_API_KEY") == "test-anthropic-key"

    provider._inject_api_key_for_model("gemini/gemini-1.5-pro", "test-gemini-key")
    assert os.environ.get("GEMINI_API_KEY") == "test-gemini-key"

    print("✓ Dynamic Model Routing and Multi-Provider key injection tests passed!")


if __name__ == "__main__":
    test_timestamp_utils()
    test_zip_loader()
    test_xml_corpus_builder()
    test_deliverable_files()
    test_dynamic_model_routing()
    print("All unit tests passed successfully!")
