import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv(
    r"M:\Files\Task\twitter_training.csv",
    header=None
)

df.columns = ['id', 'entity', 'sentiment', 'tweet']

print("Dataset Shape:", df.shape)
print(df.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_tweet'] = df['tweet'].astype(str).apply(clean_text)

sentiment_counts = df['sentiment'].value_counts()
print("\nSentiment Counts:\n", sentiment_counts)

plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 6))
sentiment_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Sentiment Percentage Distribution")
plt.ylabel("")
plt.show()

print("\n✅ Task-04 Completed Successfully!")
