import streamlit as st
import pandas as pd
import numpy as np
import difflib
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STAGE CONFIGURATION ---
st.set_page_config(
    page_title="CineMatch Premium",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN PREMIUM CSS INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top center, #1a153a 0%, #070512 100%) !important;
        color: #f1f2f6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    .hero-header {
        text-align: center;
        padding: 20px 0;
    }
    .main-title {
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 50%, #fbd38d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        margin: 0;
    }
    .sub-title {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 5px;
        font-weight: 300;
    }

    /* Layout Card Grids */
    .card-wrapper {
        background: #120e2e;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.3s ease-in-out;
        display: flex;
        flex-direction: column;
        height: 100%;
        margin-bottom: 15px;
    }
    .card-wrapper:hover {
        transform: translateY(-8px);
        border-color: #ff416c;
        box-shadow: 0 12px 24px rgba(255, 65, 108, 0.25);
    }
    .card-poster {
        width: 100%;
        aspect-ratio: 2 / 3;
        object-fit: cover;
        background-color: #1e1a3a;
    }
    .card-content {
        padding: 14px;
        background: #0e0a26;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        flex-grow: 1;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 6px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.3;
        height: 2.6rem;
    }
    .card-metrics {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
    .rating-badge { color: #ecc94b; font-weight: bold; }
    .year-badge { color: #a0aec0; }

    .spotlight-banner {
        background: rgba(18, 14, 46, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
        backdrop-filter: blur(15px);
    }
    
    .trailer-btn {
        display: inline-block;
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white !important;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        margin-top: 15px;
        transition: opacity 0.2s;
    }
    .trailer-btn:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# --- TMDb INTELLIGENCE CORE ---
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

@st.cache_data
def get_advanced_movie_details(title):
    if not TMDB_API_KEY:
        return None
    try:
        clean_title = re.sub(r'\s*\([^)]*\)', '', str(title)).strip()
        encoded_title = requests.utils.quote(clean_title)
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={encoded_title}"
        
        res = requests.get(search_url, timeout=5)
        if res.status_code == 200:
            results = res.json().get('results')
            if results:
                movie_id = results[0]['id']
                detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=videos"
                full_res = requests.get(detail_url, timeout=5)
                if full_res.status_code == 200:
                    return full_res.json()
    except Exception:
        pass
    return None

# --- DATA MODEL ENGINE ---
@st.cache_data
def load_and_sync_model():
    df = pd.read_csv('movies.csv')
    df['index'] = range(0, len(df))
        
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
        
    combined_features = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return df, similarity

try:
    df, similarity = load_and_sync_model()
    list_of_all_titles = df['title'].tolist()
except FileNotFoundError:
    st.error("❌ 'movies.csv' file missing inside execution folder structure.")
    st.stop()

# --- BRANDING ELEMENT ---
st.markdown("""
    <div class="hero-header">
        <div class="main-title">CINEMATCH</div>
        <div class="sub-title">AI Movie Recomender</div>
    </div>
""", unsafe_allow_html=True)

search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
with search_col2:
    selected_movie = st.selectbox(
        "🔍 Search Movie Matrix",
        options=[""] + list_of_all_titles,
        format_func=lambda x: "Start typing a movie name (e.g. Inception, Avatar)..." if x == "" else x,
        label_visibility="collapsed"
    )

if selected_movie:
    find_close_match = difflib.get_close_matches(selected_movie, list_of_all_titles)
    if not find_close_match:
        st.warning("⚠️ No exact matches found. Please try another search title.")
        st.stop()
        
    close_match = find_close_match[0]
    
    index_of_movie = df[df.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    movie_details = get_advanced_movie_details(close_match)
    movie_row = df[df['index'] == index_of_movie].iloc[0]
    
    fallback_img = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400"
    
    # --- INTERACTIVE SPOTLIGHT DISPLAY ---
    st.markdown('<div class="spotlight-banner">', unsafe_allow_html=True)
    h_col1, h_col2 = st.columns([1, 2.2])
    
    with h_col1:
        if movie_details and movie_details.get('poster_path'):
            st.image(f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}", use_container_width=True)
        else:
            st.image(fallback_img, use_container_width=True)
            
    with h_col2:
        st.markdown(f"<h1 style='margin-top:0; color:#fff;'>🎬 {close_match}</h1>", unsafe_allow_html=True)
        
        rating = movie_row.get('vote_average', 'N/A')
        release_date = movie_row.get('release_date', 'N/A')
        
        st.markdown(f"**⭐ Data Rating:** `{rating}/10`  |  **📅 Release:** `{release_date}`")
        st.markdown(f"**🎭 Genres:** *{movie_row.get('genres', 'N/A')}*")
        st.markdown(f"**🎬 Director:** `{movie_row.get('director', 'Unknown')}`")
        st.markdown(f"**👥 Primary Cast:** *{movie_row.get('cast', 'N/A')}*")
        
        st.markdown("<h4 style='color:#ff416c; margin-bottom:5px;'>Synopsis</h4>", unsafe_allow_html=True)
        overview = movie_row.get('overview', 'No summary loaded inside local dataset.')
        st.write(str(overview))
        
        # --- TRAILER LOADING ---
        t_key = None
        if movie_details and movie_details.get('videos', {}).get('results'):
            t_key = next((v['key'] for v in movie_details['videos']['results'] if (v['type'] == 'Trailer' or v['type'] == 'Teaser') and v['site'] == 'YouTube'), None)
            
        st.markdown("<h4 style='color:#ff416c; margin-top:20px; margin-bottom:5px;'>Official Media</h4>", unsafe_allow_html=True)
        if t_key:
            with st.expander("📺 Watch Official Trailer Inline", expanded=True):
                st.video(f"https://www.youtube.com/watch?v={t_key}")
        else:
            encoded_search = requests.utils.quote(f"{close_match} official trailer")
            youtube_search_url = f"https://www.youtube.com/results?search_query={encoded_search}"
            st.markdown(f'<a href="{youtube_search_url}" target="_blank" class="trailer-btn">🔍 Search Trailer on YouTube</a>', unsafe_allow_html=True)
                    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RECOMMENDATIONS GRID ---
    st.markdown("<br><h2 style='color: #ffffff; font-weight:800;'>🔥 Recommended For You</h2>", unsafe_allow_html=True)
    
    rec_columns = st.columns(6)
    rec_count = 0
    
    for item in sorted_similar_movies[1:]:
        if rec_count >= 6:
            break
            
        rec_index = item[0]
        rec_title = df[df.index == rec_index]['title'].values[0]
        rec_row = df[df.index == rec_index].iloc[0]
        
        rec_tmdb = get_advanced_movie_details(rec_title)
        
        if rec_tmdb and rec_tmdb.get('poster_path'):
            img_url = f"https://image.tmdb.org/t/p/w342{rec_tmdb['poster_path']}"
        else:
            img_url = fallback_img
            
        rec_year = str(rec_row.get('release_date', 'N/A'))[:4]
        rec_rating = rec_row.get('vote_average', '0.0')
        
        with rec_columns[rec_count]:
            st.markdown(f"""
                <div class="card-wrapper">
                    <img class="card-poster" src="{img_url}" />
                    <div class="card-content">
                        <div class="card-title">{rec_title}</div>
                        <div class="card-metrics">
                            <span class="rating-badge">⭐ {rec_rating}/10</span>
                            <span class="year-badge">{rec_year}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            rec_count += 1