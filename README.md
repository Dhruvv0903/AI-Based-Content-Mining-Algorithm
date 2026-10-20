# AI-Based Content Mining Algorithm

An AI-powered movie recommendation system that delivers personalized movie recommendations using **Content-Based Filtering** and **Collaborative Filtering**. Built using the **MovieLens dataset**, the application analyzes movie metadata and user rating patterns to recommend similar content through an interactive **Streamlit** interface.

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
- Scikit-learn
- SciPy

---

## Dataset

This project uses the **MovieLens Dataset** provided by GroupLens Research.

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


## Acknowledgements

Developed during my internship at **Balbhas Business Sysnomics** using the **MovieLens** dataset.
