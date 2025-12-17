from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

songs_data = pd.read_csv("spotify_millsongdata.csv")

for feature in ["artist", "song", "text"]:
    if feature in songs_data.columns:
        songs_data[feature] = songs_data[feature].fillna("")
    else:
        songs_data[feature] = ""

combined_features = (
    songs_data["artist"].astype(str)
    + " "
    + songs_data["song"].astype(str)
    + " "
    + songs_data["text"].astype(str)
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
list_of_all_songs = songs_data["song"].astype(str).to_list()


class SongRequest(BaseModel):
    song: str


@app.get("/")
def root():
    return {"message": "Song Recommendation API"}


@app.post("/recommendation")
def recommend(request: SongRequest):
    song_name = request.song

    close_matches = difflib.get_close_matches(song_name, list_of_all_songs)
    if not close_matches:
        return {
            "matched_song": None,
            "recommendations": [],
            "detail": "No similar song found in the dataset.",
        }

    close_match = close_matches[0]

    index_of_the_song = songs_data[songs_data["song"] == close_match].index[0]

    song_vector = feature_vectors[index_of_the_song]
    similarity_scores = cosine_similarity(song_vector, feature_vectors)[0]
    
    sorted_similar_songs = sorted(
        enumerate(similarity_scores), key=lambda x: x[1], reverse=True
    )

    recommendations = []

    for idx, song_info in enumerate(sorted_similar_songs[1:30]):
        index = song_info[0]
        rec_song_title = songs_data.iloc[index]["song"]
        rec_artist = songs_data.iloc[index]["artist"]
        recommendations.append(
            {
                "song": str(rec_song_title),
                "artist": str(rec_artist),
            }
        )

    return {
        "matched_song": close_match,
        "recommendations": recommendations,
    }