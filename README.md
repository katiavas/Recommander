
# 📚 Book Recommendation System

A machine learning–powered recommendation system that suggests books based on metadata and semantic embeddings. Built with LangChain, Chroma vector database, and OpenAI embeddings, this project combines classical data analysis with modern NLP techniques.

🚀 Features

Content-based recommendations using vector embeddings of book descriptions.

Integration with LangChain + Chroma for efficient similarity search.

Supports semantic querying (beyond exact keyword matches).

Exploratory Data Analysis (EDA) with visualizations to understand book metadata.

Modular pipeline for future extension to collaborative filtering or deep learning models.

📂 Dataset

Source: 7k Books with Metadata (Kaggle)

Cleaned dataset (books_cleaned.csv) with standardized categories and enriched descriptions.

🛠️ Installation
# Clone the repository
git clone https://github.com/your-username/book-recommender.git
cd book-recommender

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Dependencies include:

pandas, numpy, matplotlib, seaborn

langchain-community, langchain-openai, langchain-chroma

dotenv for API key management

🔑 Environment Setup

Create a .env file in the project root.

Add your OpenAI API key:

OPENAI_API_KEY=your_api_key_here

📖 Usage

Run the Jupyter notebooks to reproduce results:

jupyter notebook


Rec System AI.ipynb → Data loading, cleaning, and EDA

recom.ipynb → Embedding generation, vector database setup, and recommendation engine

Example:
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Generate embeddings for book descriptions
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="chroma_store", embedding_function=embeddings)

# Query similar books
results = db.similarity_search("space adventure with strong female lead")

📊 Results

Successfully builds a content-based recommender leveraging embeddings.

Demonstrates meaningful semantic similarity between books.

Example of UI : 
<img width="2537" height="911" alt="image" src="https://github.com/user-attachments/assets/ef09aa38-e4da-4200-8fe1-f494251de4f3" />

🔮 Future Work

Add collaborative filtering (matrix factorization, nearest neighbors).

Experiment with transformer-based models for richer embeddings.

Build a lightweight web app (e.g., Streamlit) for interactive recommendations.

🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

📜 License

MIT License.
