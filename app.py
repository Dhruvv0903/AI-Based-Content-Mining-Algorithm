"""Movie recommendation app with content-based and collaborative discovery."""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


@st.cache_data(show_spinner="Loading MovieLens data…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens CSV files from common project data directories."""
    roots = ("MovieLens", "data/MovieLens", "data")
    filenames = ("ratings.csv", "movies.csv", "genome-scores.csv", "genome-tags.csv")
    for root in roots:
        paths = [os.path.join(root, name) for name in filenames]
        if all(os.path.isfile(path) for path in paths):
            return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]
    raise FileNotFoundError(
        "MovieLens CSV files were not found. Place ratings.csv, movies.csv, "
        "genome-scores.csv, and genome-tags.csv in MovieLens/ or data/."
    )


def normalize_title(title: str) -> str:
    """Normalize punctuation, whitespace, and MovieLens inverted articles."""
    title = title.strip()
    match = re.match(r"^(.*),\s*(The|A|An)$", title, flags=re.IGNORECASE)
    if match:
        title = f"{match.group(2)} {match.group(1)}"
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


@st.cache_data(show_spinner="Preparing recommendation models…")
def prepare_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    scores: pd.DataFrame,
    tags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, csr_matrix, dict[int, int], dict[int, int]]:
    """Prepare movie metadata, sparse TF-IDF features, and rating matrix."""
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("")
    tag_data = scores.merge(tags, on="tagId", how="inner")
    tag_data = tag_data[tag_data["relevance"] > 0.7]
    grouped = tag_data.groupby("movieId")["tag"].apply(lambda values: " ".join(values)).reset_index()
    movies = movies.merge(grouped, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    movies["content_profile"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"]
    movies["normalized_title"] = movies["title"].map(normalize_title)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(movies["content_profile"])

    rating_frame = ratings[["movieId", "userId", "rating"]].copy()
    movie_ids = rating_frame["movieId"].drop_duplicates().tolist()
    user_ids = rating_frame["userId"].drop_duplicates().tolist()
    movie_idx = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    user_idx = {int(user_id): index for index, user_id in enumerate(user_ids)}
    rows = rating_frame["movieId"].map(movie_idx)
    cols = rating_frame["userId"].map(user_idx)
    rating_matrix = csr_matrix(
        (rating_frame["rating"], (rows, cols)),
        shape=(len(movie_ids), len(user_ids)),
    )
    return movies, rating_frame, tfidf, rating_matrix, movie_idx, user_idx


def find_movies(query: str, movies: pd.DataFrame) -> pd.DataFrame:
    """Find normalized exact or partial title matches, prioritizing exact hits."""
    normalized = normalize_title(query)
    if not normalized:
        return movies.iloc[0:0]
    matches = movies[movies["normalized_title"].str.contains(re.escape(normalized), na=False)]
    return matches.sort_values(
        by="normalized_title", key=lambda column: column.map(lambda value: (value != normalized, value))
    ).head(20)


def _movie_index(movie_id: int, movie_idx: dict[int, int]) -> int | None:
    return movie_idx.get(int(movie_id))


def recommendation_scores(
    seed: pd.Series,
    movies: pd.DataFrame,
    tfidf: Any,
    rating_matrix: csr_matrix,
    movie_idx: dict[int, int],
    limit: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return content and collaborative recommendations with similarity scores."""
    movie_position = int(seed.name)
    content_values = linear_kernel(tfidf[movie_position], tfidf).ravel()
    content_values[movie_position] = -1
    content_indices = content_values.argsort()[::-1][:limit]
    content = movies.iloc[content_indices][["movieId", "title", "genres", "tag"]].copy()
    content["score"] = content_values[content_indices]

    collab_position = _movie_index(int(seed["movieId"]), movie_idx)
    if collab_position is None:
        collab = movies.iloc[0:0][["movieId", "title", "genres", "tag"]].copy()
        collab["score"] = pd.Series(dtype=float)
        return content, collab
    similarities = cosine_similarity(rating_matrix[collab_position], rating_matrix).ravel()
    similarities[collab_position] = -1
    positions_by_id = {movie_id: pos for movie_id, pos in movie_idx.items()}
    candidates = sorted(
        ((movie_id, similarities[pos]) for movie_id, pos in positions_by_id.items() if movie_id != int(seed["movieId"])),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    score_map = {movie_id: score for movie_id, score in candidates}
    collab = movies[movies["movieId"].isin(score_map)][["movieId", "title", "genres", "tag"]].copy()
    collab["score"] = collab["movieId"].map(score_map)
    collab = collab.sort_values("score", ascending=False)
    return content, collab


def explanation(seed: pd.Series, recommendation: pd.Series) -> list[str]:
    """Return genres and genome tags shared by the seed and recommendation."""
    seed_genres = set(str(seed.get("genres", "")).split("|"))
    movie_genres = set(str(recommendation.get("genres", "")).split("|"))
    shared_genres = sorted((seed_genres & movie_genres) - {"", "(no genres listed)"})
    seed_tags = set(str(seed.get("tag", "")).split())
    movie_tags = set(str(recommendation.get("tag", "")).split())
    shared_tags = sorted(seed_tags & movie_tags)
    return (shared_genres + shared_tags)[:5]


def render_cards(frame: pd.DataFrame, ratings: pd.DataFrame, seed: pd.Series) -> None:
    """Render recommendation cards with cosine scores and shared metadata."""
    if frame.empty:
        st.info("No recommendations are available for this movie.")
        return
    rating_summary = ratings.groupby("movieId")["rating"].mean()
    for rec in frame.itertuples(index=False):
        recommendation = pd.Series(rec._asdict())
        with st.container(border=True):
            st.markdown(f"#### {recommendation['title']}")
            year_match = re.search(r"\((\d{4})\)\s*$", str(recommendation["title"]))
            st.caption(f"Release year: {year_match.group(1) if year_match else 'Not available'}")
            genres = str(recommendation.get("genres", "")).replace("|", " · ") or "Genres unavailable"
            st.write(f"**Genres:** {genres}")
            avg_rating = rating_summary.get(recommendation["movieId"])
            rating_text = f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "Not rated"
            st.write(f"**⭐ Average Movie Rating:** {rating_text}")
            st.write(f"**📊 Cosine similarity:** {float(recommendation.get('score', 0)):.1%}")
            st.markdown("**💡 Why You'll Like It**")
            reasons = explanation(seed, recommendation)
            st.caption(", ".join(reasons) if reasons else "A close match based on the movie's themes and genres.")


def analytics_page(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Show repository-wide MovieLens overview and charts."""
    st.title("📊 Analytics")
    avg_rating = float(ratings["rating"].mean()) if not ratings.empty else 0.0
    st.subheader("Dataset summary")
    with st.container(border=True):
        metrics = st.columns(5)
        for column, label, value in zip(
            metrics,
            ("Dataset", "Total movies", "Total ratings", "Total users", "Average rating"),
            ("MovieLens", f"{len(movies):,}", f"{len(ratings):,}", f"{ratings['userId'].nunique():,}", f"{avg_rating:.2f}/5"),
        ):
            column.metric(label, value)
    st.subheader("Top 10 most-rated movies")
    counts = ratings.groupby("movieId").size().nlargest(10).rename("Ratings")
    top = movies.set_index("movieId").join(counts).dropna(subset=["Ratings"]).sort_values("Ratings", ascending=False)
    st.dataframe(top[["title", "Ratings"]], use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Rating distribution")
        st.bar_chart(ratings["rating"].value_counts().sort_index(), x_label="Rating", y_label="Number of ratings", height=300)
        st.subheader("Movies released by decade")
        years = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
        decades = (pd.to_numeric(years, errors="coerce") // 10 * 10).dropna().astype(int).value_counts().sort_index()
        st.bar_chart(decades, x_label="Release decade", y_label="Number of movies", height=300)
    with right:
        genre_rows = movies.assign(genre=movies["genres"].fillna("").str.split("|" )).explode("genre")
        genre_rows = genre_rows[genre_rows["genre"].notna() & genre_rows["genre"].ne("(no genres listed)")]
        st.subheader("Genre popularity (movies)")
        st.bar_chart(genre_rows["genre"].value_counts().head(15), x_label="Genre", y_label="Number of movies", height=300)
        st.subheader("Average rating by genre")
        genre_ratings = genre_rows.merge(ratings.groupby("movieId")["rating"].mean().rename("rating"), on="movieId")
        st.bar_chart(genre_ratings.groupby("genre")["rating"].mean().sort_values(ascending=False), x_label="Genre", y_label="Average rating (out of 5)", height=300)


def genre_page(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Explore ratings and popularity for a selected movie genre."""
    st.title("🎭 Genre Explorer")
    all_genres = sorted({genre for values in movies["genres"].fillna("").str.split("|") for genre in values if genre and genre != "(no genres listed)"})
    if not all_genres:
        st.info("No genre metadata is available.")
        return
    genre = st.selectbox("Choose a genre", all_genres)
    subset = movies[movies["genres"].str.contains(rf"(?:^|\|){re.escape(genre)}(?:\||$)", na=False)].copy()
    summary = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    subset = subset.join(summary, on="movieId")
    subset["mean"] = subset["mean"].fillna(0)
    subset["count"] = subset["count"].fillna(0)
    with st.container(border=True):
        first, second = st.columns(2)
        first.metric("Movies in genre", f"{len(subset):,}")
        second.metric("Average rating", f"{subset.loc[subset['count'] > 0, 'mean'].mean():.2f}/5" if (subset["count"] > 0).any() else "N/A")
    st.subheader("Highest rated movies (at least 10 ratings)")
    st.dataframe(subset[subset["count"] >= 10].sort_values("mean", ascending=False)[["title", "mean", "count"]].head(10), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Rating histogram")
        genre_ids = subset["movieId"]
        st.bar_chart(ratings[ratings["movieId"].isin(genre_ids)]["rating"].value_counts().sort_index(), x_label="Rating", y_label="Number of ratings", height=300)
        st.subheader("Most popular (highest average rating)")
        st.dataframe(subset.sort_values("mean", ascending=False)[["title", "mean"]].head(10), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Most frequently rated")
        st.dataframe(subset.sort_values("count", ascending=False)[["title", "count"]].head(10), hide_index=True, use_container_width=True)


def recommender_page(
    movies: pd.DataFrame, ratings: pd.DataFrame, tfidf: Any, rating_matrix: csr_matrix, movie_idx: dict[int, int]
) -> None:
    """Render search, title suggestions, and independent recommendation lists."""
    st.title("🎬 Movie Recommender")
    st.caption("Find your next favorite with recommendations inspired by movie themes and viewing patterns.")
    with st.sidebar:
        st.header("Movie discovery")
        st.caption("Search the MovieLens catalog to find movies you might enjoy.")
    with st.form("movie_search"):
        query = st.text_input("Search for a movie", value=st.session_state.get("selected_title", ""), placeholder="The Matrix (1999)")
        submitted = st.form_submit_button("Get recommendations", type="primary")
    if submitted:
        matches = find_movies(query, movies)
        if matches.empty:
            st.warning("No movie titles match that search. Try another title.")
            st.session_state.pop("search_matches", None)
        else:
            st.session_state["search_matches"] = matches["movieId"].tolist()
            st.session_state["selected_title"] = query
    match_ids = st.session_state.get("search_matches", [])
    if match_ids:
        matches = movies[movies["movieId"].isin(match_ids)]
        selected_id = st.selectbox("Select a movie", matches["movieId"].tolist(), format_func=lambda movie_id: matches.loc[matches["movieId"] == movie_id, "title"].iloc[0])
        seed = movies[movies["movieId"] == selected_id].iloc[0]
        st.subheader(f"Because you selected: {seed['title']}")
        with st.spinner("Finding movies you might enjoy…"):
            content, collaborative = recommendation_scores(seed, movies, tfidf, rating_matrix, movie_idx)
        content_column, collaborative_column = st.columns(2)
        with content_column:
            st.subheader("🎭 Similar Movies")
            render_cards(content, ratings, seed)
        with collaborative_column:
            st.subheader("👥 People Also Enjoyed")
            render_cards(collaborative, ratings, seed)


def main() -> None:
    """Load the data once and route between the app's Streamlit views."""
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:3rem}</style>", unsafe_allow_html=True)
    try:
        ratings, raw_movies, scores, tags = load_data()
        movies, ratings, tfidf, rating_matrix, movie_idx, _ = prepare_data(ratings, raw_movies, scores, tags)
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(str(error))
        st.stop()
    with st.sidebar:
        st.title("🎬 MovieLens")
        page = st.radio("Navigate", ("Movie Recommender", "Analytics", "Genre Explorer"), key="page")
    if page == "Analytics":
        analytics_page(movies, ratings)
    elif page == "Genre Explorer":
        genre_page(movies, ratings)
    else:
        recommender_page(movies, ratings, tfidf, rating_matrix, movie_idx)


if __name__ == "__main__":
    main()