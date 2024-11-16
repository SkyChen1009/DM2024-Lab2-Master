ISA5810 Data Mining Lab 2

# Neural Word Embeddings and exploring Large Language Models (LLMs)

This notebook is part of a lab session focused on applying **Neural Word Embeddings** and exploring **Large Language Models (LLMs)** for various data mining tasks, including text classification and clustering.

## Table of Contents
1. Data Preparation
2. Feature Engineering
3. Model Implementation
4. Results Evaluation
5. Additional Experiments
6. Deep Learning Applications
7. Word to Vector
8. Clustering Techniques
9. High-Dimension Visualization
10. Exploring Large Language Models (LLMs)

## Objectives
The primary goal of this lab is to provide hands-on experience in:
- **Text classification** using embeddings.
- **Deep learning** methods for data mining.
- Utilizing **open-source LLMs** for advanced data tasks.
- **Clustering** and visualizing high-dimensional data.

## Dataset
The dataset used for this lab is from **[SemEval 2017 Task](https://competitions.codalab.org/competitions/16380)**. It focuses on classifying text into four distinct emotions.

![Dataset Visualization](attachment:pic0.png)

## Prerequisites
### Library Requirements
This notebook requires several libraries. Install them using the following commands:

#### Core Libraries
- **Jupyter Notebook**: `pip install jupyter`
- **Scikit Learn**: `pip install scikit-learn`
- **Pandas**: `pip install pandas`
- **Numpy**: `pip install numpy`
- **Matplotlib**: `pip install matplotlib`
- **Plotly**: `pip install plotly`
- **Seaborn**: `pip install seaborn`
- **NLTK**: `pip install nltk`
- **UMAP**: `pip install umap-learn`

#### New Libraries
- **Gensim**: `pip install gensim`
- **TensorFlow**: `pip install tensorflow tensorflow-hub`
- **Keras**: `pip install keras`
- **Ollama**: `pip install ollama`
- **LangChain**: `pip install langchain langchain_community langchain_core`
- **Beautiful Soup**: `pip install beautifulsoup4`
- **ChromaDB**: `pip install chromadb`
- **Gradio**: `pip install gradio`

#### Open-source LLMs
- Install using `ollama run llama3.2` or `ollama run llama3.2:1b` (optional).

## Key Features
- **Data Preparation**: Clean and preprocess text data for downstream tasks.
- **Feature Engineering**: Extract features using word embeddings and other methods.
- **Deep Learning Models**: Implement neural networks for text classification.
- **Clustering**: Apply unsupervised learning for grouping data.
- **Visualization**: Use dimensionality reduction techniques like UMAP for high-dimensional data.
- **RAG Systems**: Demonstrate Retrieval-Augmented Generation (RAG) using open-source LLMs.

## Getting Started
1. Clone the repository containing the notebook.
2. Install the required libraries using the commands listed above.
3. Open the notebook using Jupyter or your preferred IDE.
4. Follow the sections step-by-step, starting with data preparation.

## Results
The notebook includes evaluations of different models and techniques, highlighting the effectiveness of LLMs in tasks like text classification and clustering.

## License
This notebook is intended for educational purposes and is distributed under an open-source license.
