# AI-Based Content Mining Algorithm

An AI-powered movie recommendation system that delivers personalized movie recommendations using a hybrid approach combining **Content-Based Filtering** and **Collaborative Filtering**. Built using the **MovieLens dataset**, the application analyzes movie metadata and user rating patterns to recommend similar content through an interactive **Streamlit** interface.

---

## Features

- 🔍 Movie search with partial title matching
- 🎭 Content-Based Filtering using genres and genome tags
- 👥 Collaborative Filtering using MovieLens user ratings
- 📊 Analytics dashboard with dataset insights
- 🎬 Genre Explorer for browsing movies by category
- ⚡ Interactive Streamlit web interface
- 📈 Cosine similarity-based recommendation engine

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy
- MovieLens Dataset

---

## Dataset

This project uses the **MovieLens Latest Dataset** provided by GroupLens Research.

Required files:

- `movies.csv`
- `ratings.csv`
- `genome-tags.csv`
- `genome-scores.csv`

Download:
https://grouplens.org/datasets/movielens/

---

## Setup

1. Clone the repository.

```bash
git clone <repository-url>
cd AI-Based-Content-Mining-Algorithm
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Download the MovieLens dataset.

4. Place the CSV files inside the `MovieLens/` folder.

```
MovieLens/
├── movies.csv
├── ratings.csv
├── genome-tags.csv
└── genome-scores.csv
```

5. Launch the application.

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## Future Enhancements

- Integrate **TMDb API** for movie posters, overviews, cast information, trailers, and ratings.
- Support **OTT platform discovery** by displaying where recommended movies are currently available to stream.
- Implement **Neural Collaborative Filtering (NCF)** for improved recommendation accuracy.
- Incorporate **Natural Language Processing (NLP)** and **sentiment analysis** to enhance content understanding and personalization.
- Add user authentication, personalized watchlists, recommendation history, and cloud deployment for a complete cross-platform recommendation platform.

---

## Skills Demonstrated

- Artificial Intelligence
- Machine Learning
- Content Mining
- Recommendation Systems
- Collaborative Filtering
- Content-Based Filtering
- Information Retrieval
- Feature Engineering
- TF-IDF Vectorization
- Cosine Similarity
- Sparse Matrix Computation
- Streamlit Application Development

---

## Acknowledgements

Developed in collaboration with **Balbhas Business Sysnomics** using the **MovieLens** dataset.
