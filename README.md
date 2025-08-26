📰 Fake News Detection System






📌 Overview

The Fake News Detection System is a student project developed to address the growing challenge of fake news circulation in Nigeria.
This application uses Natural Language Processing (NLP) and Machine Learning/Deep Learning models to classify news articles as either real or fake.

By focusing on Nigerian news sources, the system aims to:

Help identify misleading or fabricated content.

Support fact-checking efforts within the local media space.

Serve as a foundation for future research into combating misinformation in Nigeria.

🖼️ Demo Screenshot
<img width="1359" height="691" alt="Screenshot (103)" src="https://github.com/user-attachments/assets/13adc58f-242f-4c6a-847d-04f106f03530" />

<img width="1340" height="691" alt="Screenshot (102)" src="https://github.com/user-attachments/assets/ebe12a5a-cacd-4324-90e0-2bcdefa0c4a9" />


🚀 Features

Preprocessing of text data (tokenization, stopword removal, stemming/lemmatization).

Training and evaluation with ML/DL models (e.g., Logistic Regression, Naive Bayes, RoBERTa).

User-friendly interface for testing news headlines/articles.

High accuracy in distinguishing between real and fake news.

🛠️ Tech Stack

Languages: Python

Libraries/Frameworks: Scikit-learn, TensorFlow/PyTorch, Pandas, NumPy, NLTK/Spacy

Frontend : Flask

Dataset: self made dataset (or your preferable dataset)

📂 Project Structure
📦 Fake-News-Detector
 ┣ 📂 data              # dataset
 ┣ 📂 models            # trained models
 ┣ 📂 notebooks         # Jupyter notebooks
 ┣ 📂 static / templates# frontend files
 ┣ 📜 app.py            # main application file
 ┣ 📜 requirements.txt  # dependencies
 ┣ 📜 README.md         # project documentation

⚙️ Installation & Usage
# Clone the repository
git clone https://github.com/CodingwithSam606/fake-news-detector.git

# Navigate into the folder
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

📊 Results

Using a fine-tuned **RoBERTa model** on a Nigerian news dataset, the system achieved:  

| Metric      | Score |
|-------------|-------|
| Accuracy    | 92%   |
| Precision   | 91%   |
| Recall      | 90%   |
| F1-score    | 91%   |

> ⚡ Note: Results are based on evaluation with a curated Nigerian news dataset.  
> Performance may vary with larger and more diverse datasets.  


🔮 Future Improvements

Improve dataset diversity

Deploy as a web application with Flask/Django/React

Extend support for multiple languages

👨‍💻 Author

Samuel Asalu

Email: samasalu10@gmail.com
