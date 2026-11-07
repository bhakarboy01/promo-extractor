import os
import json
from promo_engine import PromoEngine, load_data_files
from llm_provider import LiteLLMProvider

def test_concept_pipeline():
    print("=== Testing Concept-Driven Map-Reduce Pipeline ===")
    provider = LiteLLMProvider(model="gemini-1.5-flash")
    engine = PromoEngine(llm_provider=provider)

    # Load 15 episodes so we have enough context for distant pairings
    episodes = load_data_files("data")[:15]
    print(f"Loaded {len(episodes)} episodes.")

    def on_progress(msg, frac):
        print(f"[{int(frac*100)}%] {msg}")

    print("\n--- Running Concept-Driven Pipeline ---")
    result = engine.run_concept_driven_pipeline(
        episodes=episodes,
        progress_callback=on_progress
    )

    concepts = result["promo_concepts"]
    print(f"\nAction 2 Director Skim generated {len(concepts)} concepts:")
    for c in concepts[:3]:
        print(f"  - {c['concept_id']}: Episodes {c['episodes_involved']} | Hook: {c['target_hook'][:60]}... | Cliff: {c['target_cliffhanger'][:60]}...")

    extracted = result["extracted_promos"]
    print(f"\nAction 3 Concept Editor extracted {len(extracted)} valid promos (quota target: 8-10):")
    for p in extracted[:3]:
        print(f"  - {p['promo_id']}: Episodes {p['episodes_used']} | Dur: {p['total_duration_seconds']}s | Timestamps: {p['timestamps']}")
        assert 20 <= p['total_duration_seconds'] <= 60, f"Duration {p['total_duration_seconds']} violates 20-60s constraint!"

    ranking = result["ranking_result"]
    print("\nAction 5 Executive Ranker Top 5:")
    top_list = ranking.get("top_promos", [])
    print(f"Total ranked: {len(top_list)}")
    for tr in top_list:
        print(f"  Rank #{tr['rank']}: {tr.get('title')} (Episodes {tr.get('episodes_used')}) - Virality {tr.get('virality_score')}/100")
        print(f"    Pairing: {tr.get('pairing_analysis', '')[:100]}...")

    assert len(extracted) >= 5, "Should extract at least 5-8 promos"
    assert len(top_list) > 0, "Top 5 promos should be ranked"
    print("\n✓ Full Concept-Driven Pipeline test passed successfully!")

if __name__ == "__main__":
    test_concept_pipeline()
