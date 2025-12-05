import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# === STEP 1: Authenticate ===
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="b5c74b801ea644e28e198d3c9259a8f1",        # replace with your client ID
    client_secret="d70b239352d14031bbbeb607c8cf2093" # replace with your client secret
))

# === STEP 2: Load your dataset ===
df = pd.read_csv("278k_labelled_uri.csv")

# === STEP 3: Define label-to-emotion mapping ===
# You can adjust these based on your dataset’s meaning of 0, 1, 2, etc.
label_to_emotion = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm"
}

# === STEP 4: Search for Hindi songs for each emotion ===
emotion_to_song = {}

for label, emotion in label_to_emotion.items():
    query = f"{emotion} bollywood"
    results = sp.search(q=query, type='track', market='IN', limit=10)
    
    tracks = []
    for item in results['tracks']['items']:
        name = item['name']
        artist = item['artists'][0]['name']
        uri = item['uri']
        tracks.append((name, artist, uri))
    emotion_to_song[label] = tracks

# === STEP 5: Replace each row’s URI with a random Hindi song for that label ===
import random

def get_hindi_uri(label):
    if label in emotion_to_song and emotion_to_song[label]:
        return random.choice(emotion_to_song[label])[2]  # pick random URI
    else:
        return None

df['uri'] = df['labels'].apply(get_hindi_uri)

# === STEP 6: Save new dataset ===
df.to_csv("278k_hindi_labelled_uri.csv", index=False)

print("✅ Dataset successfully updated with Hindi/Bollywood songs!")
