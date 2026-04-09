import json
import pandas as pd
from datetime import datetime
def parse_twibot_json_custom(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rows = []
    # If this is TwiBot format, there will be 'profile', 'tweet' etc keys in dicts
    if isinstance(data, list) and len(data) > 0 and 'profile' in data[0]:
        for item in data:
            profile = item.get('profile')
            if not profile: continue
            username = profile.get('screen_name', '')
            label_val = item.get('label')
            
            # Map human/ai explicitly if label is integer, e.g. 0=human, 1=ai?
            label = "human" if label_val == 0 else "ai"
            
            tweets = item.get('tweet')
            if not tweets: continue
            
            # Use dummy timestamp if missing from json
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
print(df.head())
print("Cols:", df.columns.tolist())
