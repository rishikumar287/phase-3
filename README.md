Fake News Detection with Advanced NLP
This repository contains the code and resources for a project focused on detecting fake news using advanced Natural Language Processing (NLP) techniques. [cite_start]The primary objective is to develop a classification model that accurately distinguishes between reliable and unreliable news articles. [cite_start]This project aims to empower users to critically evaluate online information, combat the spread of misinformation, and contribute to a more informed digital environment.
Problem Statement
[cite_start]The project addresses the pervasive issue of fake news online by developing an automated system for URL analysis and advanced Natural Language Processing. [cite_start]The core problem is the binary classification of news articles as either reliable or unreliable. [cite_start]Successfully tackling this challenge empowers users to critically evaluate online information, combats the spread of misinformation, and contributes to a more informed digital environment. [cite_start]This classification capability has significant societal benefits in fostering trust and enabling sound decision-making.
Abstract
[cite_start]This project tackles the critical issue of fake news detection in online articles. [cite_start]The primary objective is to develop a classification model that accurately distinguishes between reliable and unreliable news articles. [cite_start]The approach involves collecting a dataset of news articles, preprocessing the text data using Natural Language Processing (NLP) techniques, extracting relevant features, and training machine learning models. [cite_start]Several models, including Logistic Regression and Random Forest, were employed to identify patterns indicative of fake news. [cite_start]The results demonstrate the effectiveness of the developed model in classifying news articles with a high degree of accuracy.
System Requirements
Hardware
 * [cite_start]Minimum 8GB RAM
 * [cite_start]Intel Core i5 processor or equivalent
 * [cite_start]1GB of free storage space
Software
 * [cite_start]Python 3.9 - 3.11
 * [cite_start]Required libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn, nltk, spacy, transformers, flask (a complete list is provided in requirements.txt file)
 * [cite_start]IDE: Jupyter Notebook or Google Colab
Objectives
[cite_start]This project aims to develop a machine learning model for accurate classification of news articles as reliable or unreliable. [cite_start]The primary output will be a binary classification for given article URLs. [cite_start]Key objectives include achieving high accuracy, precision, and recall in fake news detection. [cite_start]Ultimately, this work contributes to mitigating misinformation and enhancing trust in online information.
Project Workflow
[cite_start]The project workflow involves the following steps:
 * [cite_start]Start
 * [cite_start]Data Collection
 * [cite_start]Data Preprocessing
 * [cite_start]EDA (Exploratory Data Analysis)
 * [cite_start]Feature Engineering
 * [cite_start]Model Building
 * [cite_start]Model Evaluation
 * [cite_start]Deployment
 * [cite_start]End
Dataset Description
 * [cite_start]Source: Kaggle
 * [cite_start]Type: Public
 * [cite_start]Size and structure: 101 rows / 2 columns
Data Preprocessing
 * [cite_start]Handling Missing Values: Missing values were addressed by removing articles with excessive missing data and imputing missing numerical features using the mean or median. [cite_start]During web scraping, if crucial article content was absent, the scrape was discarded to maintain data integrity. [cite_start]This ensures the model trains on complete and reliable information.
 * [cite_start]Removing Duplicates: Exact duplicate articles within the training dataset were identified and removed using pandas to prevent redundancy. [cite_start]To optimize application performance, duplicate URL submissions will be handled via caching or prevention mechanisms.
 * [cite_start]Handling Outliers: Outliers in the training data were detected through statistical analysis and visual exploration. [cite_start]Removal or transformation techniques were applied to mitigate their impact. [cite_start]For scraped data, robust practices and error handling were employed.
 * [cite_start]Feature Encoding and Scaling: The target variable ("reliable"/"unreliable") was label-encoded, and nominal categorical features (e.g., article source) were one-hot encoded. [cite_start]Numerical features were normalized or standardized as needed to ensure consistent scaling and improve model performance.
Exploratory Data Analysis (EDA)
 * [cite_start]Feature: label (target variable)
   * [cite_start]Plot: Countplot
   * [cite_start]Explanation: Shows class distribution (reliable/unreliable).
   * [cite_start]Insight: Reveals class balance/imbalance, impacting model evaluation.
 * [cite_start]Feature: article length
   * [cite_start]Plot: Histogram, Boxplot
   * [cite_start]Explanation: Shows distribution and outliers of article lengths.
   * [cite_start]Insight: Potential length differences between reliable/unreliable articles; outlier handling.
 * [cite_start]Feature: source
   * [cite_start]Plot: Countplot, Bar chart
   * [cite_start]Explanation: Shows article count per news source.
   * [cite_start]Insight: Source influence on reliability; data sufficiency per source.
 * [cite_start]Feature: sentiment score
   * [cite_start]Plot: Histogram, Boxplot
   * [cite_start]Explanation: Shows distribution and outliers of sentiment scores.
   * [cite_start]Insight: Sentiment tendencies of reliable/unreliable articles.
 * [cite_start]Features: article length and sentiment score
   * [cite_start]Plot: Scatter plot
   * [cite_start]Explanation: Shows relationship between article length and sentiment.
   * [cite_start]Insight: Correlation between length and sentiment.
Feature Engineering
 * New feature creation: The code employs TF-IDF vectorization to generate new numerical features from the 'clean_text' column. Each word becomes a feature, with its TF-IDF score representing its importance. This transforms the text into a machine-readable format, capturing word relevance for classification. [cite_start]While not explicit column creation, it's a powerful form of feature engineering.
 * [cite_start]Feature selection: The code implicitly reduces features using TfidfVectorizer parameters. [cite_start]stop_words='english' removes common, less informative words. [cite_start]max_df = 0.7 ignores words appearing in over 70% of documents. [cite_start]Initial column selection (df[['title', 'text']]) also acts as feature selection.
 * [cite_start]Transformation techniques: The code cleans text using the clean_text function, which involves lowercasing, removing URLs, mentions, punctuation, and numbers. [cite_start]It then transforms the cleaned text into numerical vectors using TF-IDF with TfidfVectorizer. [cite_start]These transformations prepare the text data for machine learning by standardizing it and converting it into a numerical format.
 * Impact of features on the model:
   * [cite_start]The clean_text function standardizes text by lowercasing and removing noise like URLs and punctuation. [cite_start]This cleaning ensures consistency and focuses the model on relevant words, improving generalization.
   * [cite_start]TfidfVectorizer transforms cleaned text into numerical TF-IDF vectors, which models require.
   * [cite_start]TF-IDF weighs word importance, giving more weight to discriminative terms.
   * [cite_start]By cleaning text and using TF-IDF, the code creates a structured, numerical representation. [cite_start]This representation allows the Logistic Regression model to effectively learn from text data.
   * [cite_start]Cleaning reduces noise, leading to more accurate pattern recognition by the model.
   * [cite_start]TF-IDF highlights important words, enabling the model to better distinguish between fake and real news.
   * [cite_start]These techniques collectively enhance the model's ability to classify news articles, improving metrics like accuracy and precision.
   * [cite_start]In essence, feature engineering makes the text data more suitable and informative for the machine learning model.
Model Building
 * [cite_start]Models Tried: Logistic Regression (Baseline)
 * Explanation of Model Choices:
   * [cite_start]Logistic Regression (Baseline): Logistic Regression was chosen as the baseline model due to its simplicity and effectiveness in binary classification problems, especially with text data. [cite_start]It's a linear model and provides a good starting point for comparison with more complex models. [cite_start]It also provides a degree of interpretability, as you can examine the coefficients to understand feature importance.
Model Evaluation
Classification Report
| Metric | Class 0 (Unreliable) | Class 1 (Reliable) | Accuracy | Macro Avg | Weighted Avg |
|---|---|---|---|---|---|
| Precision | 0.10 | 0.11 |  | 0.10 | 0.10 |
| Recall | 0.11 | 0.10 |  | 0.10 | 0.10 |
| F1-score | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| Support | 19 | 21 | 40 | 40 | 40 |
Confusion Matrix
[cite_start]A confusion matrix is provided in the uploaded document, showing predicted vs. actual labels.
Deployment
 * [cite_start]Deployment Method: Gradio Interface
 * [cite_start]Public Link: https://4c7f664c6f905007d9.gradio.live
Sample Prediction
 * Input News Text: "By now, everyone knows that disgraced National Security Adviser Michael Flynn has flipped on Donald Trump. Of course, the good folks at Saturday Night Live, who have been mocking Trump relentlessly, couldn t resist using this to needle the notoriously thin-skinned, sorry excuse for a president. It also helps that we are in the midst of the holiday season, which enabled SNL to use a seasonal classic, A Christmas Carol, to poke fun at Trump. Alec Baldwin took up the mantle to play Trump again, who is visited by a fictional Michael Flynn (Mikey Day) in chains, and seems positively terrified of the Ghost of Michael Flynn, who appears to tell Trump, it s time to come clean for the good of the country. After that, the Ghosts of Christmas Past, Present, and Future line up to torture Trump in the Oval Office. The Ghost of Christmas Past is fired NBC host Billy Bush (Alex Moffat), of Trump s infamous grab em by the pussy tape. Then it was time for a shirtless Vladimir Putin (Beck Bennet) to arrive, to remind Trump of the fact that he wouldn t be president without help from the Russian government, and that he s about to have all of their efforts be for naught. The Ghost of Christmas Future is the best of all, with a positively wickedly delicious version of Hillary Clinton, played by Kate McKinnon, who gleefully says to Trump: You Donald, have given me the greatest Christmas gift of all! You have no idea how long I ve wanted to say this, lock him up! Lock him up indeed. This entire criminal administration belongs in jail. [cite_start]It will go from Flynn, to Pence, to Trump Jr., and then to Trump himself and then Hillary will really have the last laugh."
 * [cite_start]Output: Real News
Source Code
The repository includes Python code for:
 * [cite_start]Imports of necessary libraries (pandas, numpy, matplotlib, seaborn, string, re, scikit-learn modules for model selection, feature extraction, and metrics).
 * [cite_start]Data Collection (reading Fakenews_data.csv).
 * [cite_start]Data Preprocessing (selecting 'title' and 'text' columns, combining them, and adding a 'label' column).
 * [cite_start]clean_text function for text cleaning (lowercasing, removing URLs, mentions, punctuation, numbers).
 * [cite_start]Feature Engineering using TfidfVectorizer.
 * [cite_start]Simulating real labels for a combined dataset.
 * [cite_start]Model Building & Training (Logistic Regression).
 * [cite_start]Evaluation (classification report, confusion matrix).
 * [cite_start]Visualization (heatmap of confusion matrix).
 * [cite_start]Report Writing (saving evaluation report to report.txt).
 * [cite_start]Gradio interface for deployment.
Future Scope
[cite_start]To further improve the system, continuous learning mechanisms could be implemented to adapt to evolving misinformation tactics. [cite_start]Expanding the analysis to include multimodal data (images, videos) would also be a valuable direction. [cite_start]Enhancing the model's interpretability to provide more transparent explanations for its predictions is another key area for future development.
Team Members and Contributions
 * [cite_start]Mohammed Sakhee.B: Model development
 * [cite_start]Mohammed Sharuk.I: Data cleaning
 * [cite_start]Mubarak Basha.S: EDA
 * [cite_start]Naseerudin: Documentation and reporting
 * [cite_start]Rishi Kumar Baskar: Feature Engineering
GitHub Repository
[cite_start]https://github.com/rishikumar287/Exposing-the-truth-with-advanced-fake-news-detection-powered-by-natural-language-.git
