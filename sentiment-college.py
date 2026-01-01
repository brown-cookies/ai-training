from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re


STOP = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'}


def clean(s):
    """Clean text: lowercase, remove URLs/mentions/punctuation, strip whitespace"""
    s = s.lower()
    s = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def prep(s):
    """Preprocess: clean and remove stop words"""
    return ' '.join(w for w in clean(s).split() if w not in STOP)


def load_data():
    """Generate sample dataset"""
    pos = [
        "I love this product! It's amazing and works perfectly.",
        "Great experience! Highly recommend to everyone.",
        "Fantastic service, very happy with my purchase.",
        "This is the best thing I've ever bought. Absolutely love it!",
        "Exceeded my expectations. Really impressed.",
        "Wonderful quality and fast shipping. Will buy again!",
        "Amazing features and great value for money.",
        "Very satisfied with this purchase. It's awesome!",
        "Excellent product, does exactly what it says.",
        "Love it! Best decision I made this year.",
    ]
    neg = [
        "Terrible product. Complete waste of money.",
        "Very disappointed. Does not work as advertised.",
        "Poor quality. Broke after just one use.",
        "Worst purchase ever. Would not recommend.",
        "Horrible experience. Customer service was rude.",
        "This is garbage. Save your money and buy something else.",
        "Completely unsatisfied. Returning immediately.",
        "Don't buy this. It's a total scam.",
        "Awful quality. Not worth the price at all.",
        "Regret buying this. Such a disappointment.",
    ]
    X = (pos + neg) * 10
    y = ([1] * 10 + [0] * 10) * 10
    return X, y


def train_model(X, y):
    """Train sentiment classifier"""
    X = [prep(s) for s in X]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr_vec, y_tr)

    y_pred = clf.predict(X_te_vec)
    acc = accuracy_score(y_te, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=['NEG', 'POS']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

    return vec, clf


def predict(vec, clf, text):
    """Predict sentiment for single text"""
    processed = prep(text)
    feat = vec.transform([processed])
    pred = clf.predict(feat)[0]
    conf = clf.predict_proba(feat)[0][pred]
    return pred, conf


def main():
    print("="*50)
    print("SENTIMENT ANALYSIS")
    print("="*50 + "\n")

    X, y = load_data()
    print(f"Dataset: {len(X)} samples ({sum(y)} pos, {len(y)-sum(y)} neg)\n")

    vec, clf = train_model(X, y)

    tests = [
        "This is absolutely wonderful! I'm so happy!",
        "Terrible experience. Never buying again.",
        "It's okay, nothing special but does the job.",
        "Best product ever! Five stars!",
        "Product can be improved more"
    ]

    print("\n" + "="*50)
    print("TEST PREDICTIONS")
    print("="*50)

    for t in tests:
        p, c = predict(vec, clf, t)
        label = "POS" if p == 1 else "NEG"
        print(f"\n{t}")
        print(f"→ {label} ({c:.1%})")

    print("\n" + "="*50)


if __name__ == "__main__":
    main()
