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
    assert os.path.exists("extracted_promos.json"), "extracted_promos.json must exist"
    assert os.path.exists("ranked_promos.json"), "ranked_promos.json must exist"
    
    with open("extracted_promos.json", "r", encoding="utf-8") as f:
        ext = json.load(f)
    with open("ranked_promos.json", "r", encoding="utf-8") as f:
        rnk = json.load(f)
        
    assert len(ext) >= 8, f"Must meet minimum 8-10 promo quota, found {len(ext)}"
    assert "top_promos" in rnk, "ranked_promos.json must contain top_promos"
    assert len(rnk["top_promos"]) > 0, "top_promos must not be empty"
    print(f"✓ Deliverables verified! extracted_promos.json ({len(ext)} promos), ranked_promos.json ({len(rnk['top_promos'])} top promos).")

if __name__ == "__main__":
    test_timestamp_utils()
    test_zip_loader()
    test_xml_corpus_builder()
    test_deliverable_files()
    print("All unit tests passed successfully!")
