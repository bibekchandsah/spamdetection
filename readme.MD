# Spam Email Detection

This project is a machine learning-based spam email detection system implemented in Python. It uses the Naive Bayes algorithm to classify emails as spam or ham (not spam).

## Dataset

The dataset used is `spam.csv`, which contains labeled email messages. Each message is labeled as either 'spam' or 'ham'.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- wordcloud
- seaborn
- ipython
- textblob
- nltk
- xgboost
- joblib 

You can install the required packages using:

```bash
pip install pandas scikit-learn matplotlib wordcloud
```
or,
```bash
pip install -r requirements.txt
``` 

## Project Structure

- `spam.csv`: The dataset file containing email messages and their labels.
- `spam.ipynb`: The main script to train the model and make predictions.
- `app.py`: The app script is streamlit based UI
- `spam_detector_model.pkl`: Trained Model
- `tfidf_vectorizer.pkl`: Vectorized Model
- `README.md`: This file.

## Usage

1. **Train the Model**: Run the `spam.ipynb` script to train the spam detection model using the dataset.

   ```bash
   streamlit run app.py
   ```

2. **Predict Spam or Ham**: Use the trained model to predict whether a new message is spam or ham. Modify the `test_message` variable in `spam.ipynb` to test different messages.

3. **Visualizations**: The script also generates visualizations such as a pie chart for spam vs. ham distribution and a word cloud for spam words.

## Features

- **Spam vs. Ham Distribution**: Visualizes the proportion of spam and ham messages in the dataset.
- **Word Cloud**: Generates a word cloud for the most common words in spam messages.
- **Statistical Analysis**: Provides statistical data on the number of characters, words, and sentences in spam and ham messages.

## License

This project is licensed under the MIT License.
