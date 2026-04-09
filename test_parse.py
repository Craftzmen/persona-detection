import json
import pandas as pd
from datetime import datetime

def parse_twibot_json_custom(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    if isinstance(data, list) and len(data) > 0 and 'profile' in data[0]:
        for item in data:
            profile = item.get('profile')
            if not profile: continue
            username = profile.get('screen_name', '')
            
            label_val = item.get('label')
            label = "human" if str(label_val) == "0" else ("ai" if str(label_val) == "1" else "human")
            
            tweets = item.get('tweet')
            if not tweets: continue
            for t in tweets:
                rows.append({
                    "username": username,
                    "tweet_text": t,
                    "timestamp": datetime.now().isoformat(),
                    "label": label
                })
        return pd.DataFrame(rows)
    return pd.DataFrame()

df = parse_twibot_json_custom("data/raw/TwiBot-20_sample.json")
print("Parsed TwiBot-20 shape:", df.shape)
