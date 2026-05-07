import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#PAGE CONFIG
st.set_page_config(
    page_title="Spotify Recommendation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CUSTOM CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stMetric {
        background-color: #1DB954;
        padding: 10px;
        border-radius: 10px;
    }
    .song-card {
        background-color: #181818;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#LOAD DATA
def load_data():
    df = pd.read_csv("SpotifyFeatures.csv")
    df = df.drop_duplicates()
    return df
songs=load_data()

#SPO API
client_credentials_manager = SpotifyClientCredentials(
    client_id=st.secrets["42b1e14df1c049419d396018ce35e138"],
    client_secret=st.secrets["19cec02f632c4244a7fc801905474f48"]
)

sp = spotipy.Spotify(
    auth_manager=client_credentials_manager
)

#SPO METADATA
@st.cache_data(show_spinner=False)
def get_track_metadata(track_name, artist_name):
    try:
        query = f"track:{track_name} artist:{artist_name}"
        result = sp.search(q=query, type="track", limit=1)
        items = result["tracks"]["items"]
        
        if len(items) == 0:
            return None
        track = items[0]
        return {
            "cover": track["album"]["images"][0]["url"],
            "preview": track["preview_url"],
            "spotify_url": track["external_urls"]["spotify"],
            "album": track["album"]["name"]
        }
    except:
        return None
    
#SIDEBAR
st.sidebar.title("Spotify Dashboard")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Rule-Based Recommendation",
        "Content-Based Recommendation",
        "Hybrid Recommendation",
        "Music Analytics"
    ]
)

#OERVIEW
if menu == "Overview":
    st.title("Spotify Recommendation Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Songs", len(songs))
    col2.metric("Total Artists", songs['artist_name'].nunique())
    col3.metric("Total Genres", songs['genre'].nunique())
    col4.metric("Average Popularity", round(songs['popularity'].mean(),2))

    st.divider()
    st.subheader("Top Genres")
    genre_count = songs['genre'].value_counts().head(10)

    fig = px.bar(
        genre_count,
        x=genre_count.index,
        y=genre_count.values,
        labels={'x':'Genre','y':'Count'}
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Popularity Distribution")
    fig2 = px.histogram(
        songs,
        x='popularity',
        nbins=30
    )
  
    st.plotly_chart(fig2, use_container_width=True)

#RULE BASED RECOM
elif menu == "Rule-Based Recommendation":
    st.title("Rule-Based Recommendation")
    st.write("Recommendation berdasarkan aturan IF-THEN")
    mood = st.selectbox(
        "Select Mood",
        ["Party", "Chill", "Happy", "Sad"]
    )

    if mood == "Party":
        recs = songs[
            (songs['danceability'] > 0.7) &
            (songs['energy'] > 0.7)
        ].head(10)
    elif mood == "Chill":
        recs = songs[
            (songs['acousticness'] > 0.6)
        ].head(10)
    elif mood == "Happy":
        recs = songs[
            (songs['valence'] > 0.7)
        ].head(10)
    else:
        recs = songs[
            (songs['valence'] < 0.3)
        ].head(10)

    cols = st.columns(5)
    for idx, (_, row) in enumerate(recs.iterrows()):
        meta = get_track_metadata(row['track_name'], row['artist_name'])
        with cols[idx % 5]:
            if meta:
                st.image(meta['cover'])
            st.markdown(f"**{row['track_name']}**")
            st.write(row['artist_name'])
            st.write(f"Popularity: {row['popularity']}")
            if meta and meta['preview']:
                st.audio(meta['preview'])

#CONTENT BASED RECOM
elif menu == "Content-Based Recommendation":
    st.title("Content-Based Recommendation")
    selected_song = st.selectbox(
        "Select Song",
        songs['track_name'].unique()
    )
    features = [
        'danceability',
        'energy',
        'acousticness',
        'valence',
        'tempo'
    ]

    model_data = songs[features]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(model_data)
  
    similarity = cosine_similarity(scaled_data)
    song_index = songs[songs['track_name'] == selected_song].index[0]
    similarity_scores = list(enumerate(similarity[song_index]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_songs = similarity_scores[1:11]
    st.subheader("Recommended Songs")
    cols = st.columns(5)
    for idx, (i, score) in enumerate(top_songs):
        row = songs.iloc[i]
        meta = get_track_metadata(
            row['track_name'],
            row['artist_name']
        )

        with cols[idx % 5]:
            if meta:
                st.image(meta['cover'])
            st.markdown(f"**{row['track_name']}**")
            st.write(row['artist_name'])
            st.write(f"Similarity: {round(score,3)}")

            if meta and meta['preview']:
                st.audio(meta['preview'])

            if meta:
                st.markdown(
                    f"[Open Spotify]({meta['spotify_url']})"
                )

#HYBRID RECOM
elif menu == "Hybrid Recommendation":
    st.title("Hybrid Recommendation")
    st.write("Gabungan similarity + popularity")
    alpha = st.slider(
        "Similarity Weight",
        0.0,
        1.0,
        0.7
    )
    selected_song = st.selectbox(
        "Select Song",
        songs['track_name'].unique(),
        key='hybrid'
    )
    features = [
        'danceability',
        'energy',
        'acousticness',
        'valence',
        'tempo'
    ]

    model_data = songs[features]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(model_data)

    similarity = cosine_similarity(scaled_data)
    song_index = songs[songs['track_name'] == selected_song].index[0]
    similarity_scores = similarity[song_index]
    popularity_scaled = MinMaxScaler().fit_transform(
        songs[['popularity']]
    ).flatten()

    final_score = (
        alpha * similarity_scores +
        (1-alpha) * popularity_scaled
    )
    songs['final_score'] = final_score
    recs = songs.sort_values(
        by='final_score',
        ascending=False
    ).iloc[1:11]

    cols = st.columns(5)
    for idx, (_, row) in enumerate(recs.iterrows()):
        meta = get_track_metadata(
            row['track_name'],
            row['artist_name']
        )
        with cols[idx % 5]:
            if meta:
                st.image(meta['cover'])
            st.markdown(f"**{row['track_name']}**")
            st.write(row['artist_name'])
            st.write(f"Hybrid Score: {round(row['final_score'],3)}")
            if meta and meta['preview']:
                st.audio(meta['preview'])

#MUSIC ANALYSIS
elif menu == "Music Analytics":
    st.title("Music Analytics")
    st.subheader("Energy vs Danceability")
    fig = px.scatter(
        songs.sample(2000),
        x='energy',
        y='danceability',
        color='genre',
        hover_data=['track_name']
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Radar Chart")
    song_name = st.selectbox(
        "Select Song",
        songs['track_name'].unique(),
        key='radar'
    )
    row = songs[songs['track_name'] == song_name].iloc[0]
    radar_features = [
        'danceability',
        'energy',
        'acousticness',
        'valence'
    ]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=[row[f] for f in radar_features],
        theta=radar_features,
        fill='toself',
        name=song_name
    ))
    st.plotly_chart(fig2, use_container_width=True)
