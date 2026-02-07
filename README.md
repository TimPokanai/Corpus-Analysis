# Corpus Analysis

## 1. What you will find in this repository

After extracting the ZIP file, your directory should look like this:

Corpus-Analysis/ <br>
├── data/ <br>
├── src/ <br>
├── .gitignore <br>
└── README.md <br>

## 2. Directory and file descriptions

### data/raw

- The data/raw directory contains the raw csv file containing the BBC News Articles dataset, which must be present for the code to be functional

### src

- The script load_data.py contains code that loads the raw csv file, and decomposes it into 5 categories, with each category having its own representative documents

- The script preprocess.py contains the logic for preprocessing the dataset using lowercasing, tokenization, stop word removal, and stemming (optional)

- The script build_bow.py contains logic that builds bag of words representations of the corpus, using either a count representation or a binary representation

- The script naive_bayes.py contains the mathematical logic of the naive bayes algorithm, used to obtain the top k words of a category based on LLR

- The topic_modelling script contains LDA modelling logic for experimentation

- main.py contains all steps of experimentation with different processing, bag of word representations, and models

## 3. Setting up the environment

### Step 1: Create and setup a new Conda environment

- Open a terminal, cd into the directory "Corpus-Analysis", and run: `conda create -n MYENV python=3.14.2`

- Activate the environment: `conda activate MYENV`

### Step 2: Install required packages

- Install the required dependencies using conda: `conda install -n MYENV numpy pandas nltk scipy gensim`

## 4. Running the experiments

- First, make sure you have activated your conda environment and installed all required dependencies on that conda environment

- Open a terminal, cd into the directory "Corpus-Analysis/src", and run: `python main.py`

- Observe the results in your terminal, the program takes about a minute to complete so give it some time
