import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords (run once)
nltk.download('stopwords')

# Step 1: Create dataset
data = {
    'label': ['phishing', 'phishing', 'phishing', 'legitimate', 'legitimate', 'legitimate', 'phishing', 'legitimate'],
    'message': [
        "Your bank account has been suspended. Click here to verify your details!",
        "Urgent: Update your payment information now or your account will be locked.",
        "Congratulations! You've won a $1000 gift card. Redeem now!",
        "Your statement for June is ready to view.",
        "Reminder: Your appointment is scheduled for tomorrow.",
        "Your bank transaction was successful.",
        "Alert: Unauthorized login attempt detected. Verify your identity here.",
        "Thanks for your payment. Your balance is updated."
    ]
}

df = pd.DataFrame(data)

# Step 2: Clean text function
def clean_text(text):
    text = ''.join([c for c in text if c not in string.punctuation])  # Remove punctuation
    text = text.lower()  # Lowercase
    stop_words = set(stopwords.words('english'))  # Stopwords set
    tokens = text.split()  # Tokenize
    filtered = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(filtered)

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'legitimate': 0, 'phishing': 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.25, random_state=42
)

# Step 4: Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict function
def predict_phishing(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return "Phishing" if pred == 1 else "Legitimate"

# Step 8: Test new messages
test_messages = [
    "Your bank account will be closed unless you update your details immediately.",
    "Monthly bank statement is now available online.",
    "You have received a bonus! Click to claim your reward.",
    "Payment received. Thank you for your business."
]

for msg in test_messages:
    print(f"Message: {msg}\nPrediction: {predict_phishing(msg)}\n")
  
