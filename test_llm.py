import os
import json
from llm_provider import LiteLLMProvider
from promo_engine import PromoEngine, load_data_files

def test_single_episode_extraction():
    provider = LiteLLMProvider(model="gemini-1.5-flash")
    engine = PromoEngine(llm_provider=provider)
    
    episodes = load_data_files("data")
    ep_num, path, content = episodes[0]
    
    print(f"Testing extraction on Episode {ep_num} ({os.path.basename(path)})...")
    result = engine.process_episode(
        episode_id=ep_num,
        script_text=content[:2500], # Pass first chunk for quick test
        current_memory=""
    )
    
    print("Success! Result JSON:")
    print(json.dumps(result, indent=2))
    assert "episode_id" in result
    assert "potential_promos" in result
    assert "new_episode_summary" in result
    print("All assertions passed!")

if __name__ == "__main__":
    test_single_episode_extraction()

