from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

print("Building Robust ML Dataset (60+ samples)...")

# 1. Expanded Dataset for better training/evaluation
data = [
    # Career (20 samples)
    ("Should I quit my stable job for a startup offering equity?", "Career", "High", "Anxious"),
    ("Thinking about asking my boss for a 20% raise tomorrow.", "Career", "Medium", "Confident"),
    ("Should I go back to grad school or keep gaining work experience?", "Career", "Medium", "Conflicted"),
    ("Got an offer from a competitor, deciding if I should leverage it or leave.", "Career", "High", "Excited"),
    ("I want to pivot from marketing to software engineering.", "Career", "High", "Determined"),
    ("Should I take a lateral move to a different team with a better manager?", "Career", "Low", "Optimistic"),
    ("Considering starting a freelance business and leaving my 9-5.", "Career", "High", "Nervous"),
    ("Should I accept a lower salary for a fully remote position?", "Career", "Medium", "Analytical"),
    ("Thinking of reporting a toxic coworker to HR.", "Career", "High", "Stressed"),
    ("Should I stay for my bonus or leave for a new opportunity now?", "Career", "Medium", "Calculated"),
    ("Is it worth getting a PMP certification to move into management?", "Career", "Low", "Ambitious"),
    ("Should I negotiate my starting salary at a new firm?", "Career", "Medium", "Nervous"),
    ("Considering a career break for 6 months to travel.", "Career", "High", "Adventurous"),
    ("Should I apply for a role I'm only 60% qualified for?", "Career", "Medium", "Hopeful"),
    ("Deciding whether to take an internal promotion that requires relocation.", "Career", "High", "Conflicted"),
    ("Should I join an early-stage startup or a Big Tech company?", "Career", "High", "Curious"),
    ("Thinking of starting an MBA program while working full-time.", "Career", "Medium", "Determined"),
    ("Should I accept a contract-to-hire position?", "Career", "Medium", "Cautious"),
    ("Is it better to specialize in AI or stay a generalist dev?", "Career", "Low", "Analytical"),
    ("Should I retire early or keep working for three more years?", "Career", "Medium", "Relieved"),

    # Relationships / Social (15 samples)
    ("Should I move in with my partner after 6 months of dating?", "Relationship", "High", "Vulnerable"),
    ("Thinking about confronting my friend about a lie they told.", "Relationship", "Medium", "Frustrated"),
    ("Should I plan a solo backpacking trip across Europe or go with friends?", "Relationship", "Low", "Excited"),
    ("Deciding whether to invite my toxic family member to my wedding.", "Relationship", "High", "Stressed"),
    ("Want to ask out my coworker but worried about office dynamics.", "Relationship", "High", "Anxious"),
    ("Should I end my long-distance relationship or move to their city?", "Relationship", "High", "Sad"),
    ("Thinking of telling my best friend I have feelings for them.", "Relationship", "High", "Vulnerable"),
    ("Should I get a roommate to save money or live alone for peace?", "Relationship", "Low", "Analytical"),
    ("Deciding if I should forgive my partner after a major argument.", "Relationship", "Medium", "Hopeful"),
    ("Should I cut off contact with a negative friend?", "Relationship", "Medium", "Determined"),
    ("Thinking about proposing to my significant other during the holidays.", "Relationship", "High", "Excited"),
    ("Should we start couples counseling before getting engaged?", "Relationship", "Low", "Proactive"),
    ("Considering adopting a cat with my girlfriend.", "Relationship", "Low", "Happy"),
    ("Should I tell my parents I'm moving across the country?", "Relationship", "Medium", "Anxious"),
    ("Deciding whether to attend a high school reunion.", "Relationship", "Low", "Reluctant"),

    # Finance / Investing (15 samples)
    ("Should I put my entire savings into a high-yield crypto fund?", "Finance", "High", "Reckless"),
    ("Deciding between paying off my student loans or investing in the S&P 500.", "Finance", "Medium", "Analytical"),
    ("Should I buy a house now with high interest rates or keep renting?", "Finance", "High", "Overwhelmed"),
    ("Thinking of selling my car to save money and just biking everywhere.", "Finance", "Medium", "Motivated"),
    ("Should I max out my 401k this year even if it means strict budgeting?", "Finance", "Low", "Disciplined"),
    ("Considering taking out a personal loan to fund a risky side hustle.", "Finance", "High", "Desperate"),
    ("Should I buy NVDA stock after this massive rally?", "Finance", "High", "Greedy"),
    ("Is it time to start a 529 plan for my newborn child?", "Finance", "Low", "Responsible"),
    ("Should I refinance my mortgage or stay with my current rate?", "Finance", "Medium", "Calculated"),
    ("Thinking about buying a luxury watch as an investment.", "Finance", "Medium", "Vanity"),
    ("Should I loan money to a family member in need?", "Finance", "High", "Guilty"),
    ("Deciding between a robo-advisor or an active wealth manager.", "Finance", "Low", "Analytical"),
    ("Should I cancel my expensive gym membership to save $200 a month?", "Finance", "Low", "Frugal"),
    ("Considering a high-stakes poker game to win back losses.", "Finance", "High", "Dangerous"),
    ("Should I lease a new Tesla or buy a used Toyota?", "Finance", "Medium", "Practical"),

    # Health / Lifestyle (15 samples)
    ("Should I hire a personal trainer or just do at-home workouts?", "Lifestyle", "Low", "Motivated"),
    ("Deciding whether to adopt a large high-energy dog while living in a small apartment.", "Lifestyle", "High", "Conflicted"),
    ("Should I invest in laser eye surgery or stick with contacts?", "Lifestyle", "Medium", "Hopeful"),
    ("Thinking about moving to a completely new city where I know no one.", "Lifestyle", "High", "Adventurous"),
    ("Should I start meal prepping on Sundays to save time and eat healthier?", "Lifestyle", "Low", "Productive"),
    ("Considering a marathon training plan despite a minor knee injury.", "Lifestyle", "Medium", "Stubborn"),
    ("Should I switch to a plant-based diet for health reasons?", "Lifestyle", "Low", "Curious"),
    ("Thinking about getting a tattoo on my forearm.", "Lifestyle", "Medium", "Nervous"),
    ("Should I take a digital detox for 30 days?", "Lifestyle", "Medium", "Determined"),
    ("Deciding whether to join a local soccer league.", "Lifestyle", "Low", "Social"),
    ("Should I undergo an elective surgery to improve aesthetics?", "Lifestyle", "High", "Vain"),
    ("Thinking of learning to sail this summer.", "Lifestyle", "Low", "Excited"),
    ("Should I sleep 8 hours every night if it means waking up at 5am?", "Lifestyle", "Low", "Disciplined"),
    ("Considering buying an expensive ergonomic chair for my home office.", "Lifestyle", "Low", "Comfortable"),
    ("Should I join a meditation retreat to handle work stress?", "Lifestyle", "Medium", "Exhausted")
]

df = pd.DataFrame(data, columns=['text', 'category', 'risk', 'sentiment'])

# ── Sentiment Clustering ──────────────────────────────────────────────────
# Cluster many unique sentiments into stable categories for better ML learning
sentiment_map = {
    'Anxious': 'High Stress', 'Nervous': 'High Stress', 'Stressed': 'High Stress', 
    'Overwhelmed': 'High Stress', 'Desperate': 'High Stress', 'Exhausted': 'High Stress',
    'Analytical': 'Logic/Analytical', 'Calculated': 'Logic/Analytical', 'Disciplined': 'Logic/Analytical',
    'Cautious': 'Logic/Analytical', 'Proactive': 'Logic/Analytical', 'Responsible': 'Logic/Analytical',
    'Practical': 'Logic/Analytical', 'Frugal': 'Logic/Analytical', 'Comfortable': 'Logic/Analytical',
    'Confident': 'Positive/Growth', 'Excited': 'Positive/Growth', 'Optimistic': 'Positive/Growth',
    'Hopeful': 'Positive/Growth', 'Happy': 'Positive/Growth', 'Adventurous': 'Positive/Growth',
    'Motivated': 'Positive/Growth', 'Ambitious': 'Positive/Growth', 'Productive': 'Positive/Growth',
    'Relieved': 'Positive/Growth', 'Curious': 'Positive/Growth', 'Reluctant': 'Positive/Growth',
    'Social': 'Positive/Growth',
    'Conflicted': 'Conflicted/Vulnerable', 'Vulnerable': 'Conflicted/Vulnerable', 
    'Frustrated': 'Conflicted/Vulnerable', 'Sad': 'Conflicted/Vulnerable', 'Guilty': 'Conflicted/Vulnerable',
    'Stubborn': 'Conflicted/Vulnerable',
    'Reckless': 'Impulse/Risk', 'Greedy': 'Impulse/Risk', 'Dangerous': 'Impulse/Risk', 
    'Vanity': 'Impulse/Risk', 'Vain': 'Impulse/Risk'
}
df['sentiment'] = df['sentiment'].map(sentiment_map).fillna('Neutral/Logic')

# Function to train and evaluate a model
def train_and_eval(target_col, model_name):
    print(f"\n--- Training {model_name} ({target_col}) ---")
    
    # Define Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(random_state=42, C=1.0, class_weight='balanced'))
    ])

    # 5-Fold Cross-Validation (Robust Metrics)
    cv_strat = df[target_col] if target_col != 'sentiment' else None
    cv_scores = cross_val_score(pipeline, df['text'], df[target_col], cv=5)
    
    # Split for Final Holdout Evaluation
    strat = df[target_col] if target_col != 'sentiment' else None
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df[target_col], test_size=0.2, random_state=42, stratify=strat
    )
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"5-Fold CV Mean Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})")
    print(f"Final Holdout Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(report)
    
    # Save Model
    joblib.dump(pipeline, os.path.join(os.path.dirname(__file__), f'model_{target_col}.pkl'))
    
    # Save Report to file for documentation
    report_path = os.path.join(os.path.dirname(__file__), f'report_{target_col}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name} ({target_col})\n")
        f.write(f"5-Fold CV Mean Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})\n")
        f.write(f"Final Holdout Accuracy: {acc:.2f}\n")
        f.write("-" * 30 + "\n")
        f.write(report)

# Train all models
os.makedirs(os.path.dirname(__file__), exist_ok=True)
train_and_eval('category', 'Decision Domain Classifier')
train_and_eval('risk', 'Decision Risk Estimator')
train_and_eval('sentiment', 'User Sentiment Analysis')

print("\nAll models trained, evaluated, and saved with metric reports!")
