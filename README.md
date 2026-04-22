# Decision Assistant AI 🧠

**Decision Assistant AI** is a professional-grade strategic reasoning platform designed to break down complex dilemmas into structured, actionable insights. It utilizes a **Hybrid Intelligence Pipeline**—combining local Machine Learning classification with deep Cloud AI reasoning.

## ✨ Features

- **Double-Layer Reasoning:** 
  - **Local ML Layer:** Uses Scikit-Learn models to instantly classify decision Categorization, Risk, and Sentiment.
  - **Cloud Reasoning Layer:** Generates high-level strategic analyses via Google Gemini.
- **Multi-Perspective dashboards:** Analysis through three distinct lenses: **Practical**, **Optimistic**, and **Worst-Case Scenario**.
- **Self-Healing AI Architecture:** Custom model-cascading logic handles API rate limits by switching models automatically.
- **Dynamic Data Visualization:** Animated telemetry bars showing Logical Depth, Emotional Weight, and AI Confidence.
- **Interactive History:** Archive of past decisions with local-timezone tracking and quick-reload capability.
- **Premium UX:** Liquid Glassmorphism design system built with Vanilla CSS and GSAP animations.

## 💻 Tech Stack

### Backend
- **FastAPI / Python**: Asynchronous REST API.
- **Scikit-Learn**: Local ML pipeline (TF-IDF + Logistic Regression).
- **Google GenAI API**: LLM Reasoning engine.
- **Pydantic**: Strict data validation.
- **Joblib**: Model serialization.

### Frontend
- **Vanilla JS (ES6+)**
- **Modern CSS**: Custom "Liquid Glass" design system.
- **GSAP (GreenSock)**: Orchestrated UI transitions.
- **Lucide Icons**: Intuitive dashboards.

## 📊 Machine Learning Pipeline

The project includes a robust ML framework:
- **Pipeline:** Text ➟ TF-IDF Vectorization ➟ Logistic Regression Classifier.
- **Validation:** 5-Fold Cross-Validation for statistical reliability.
- **Evaluation:** Detailed precision, recall, and F1 reports generated per training run.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### Installation

1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd decision-assistant-ai
   ```

2. **Setup Environment:**
   Create a `.env` file in the root:
   ```env
   GEMINI_API_KEY=your_key_here
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML Models:**
   ```bash
   python backend/ml/train.py
   ```

5. **Run the Server:**
   ```bash
   python -m uvicorn backend.main:app --port 8001 --reload
   ```

6. **Open the App:**
   Simply open `app/index.html` in your browser.

## ⚖️ License
MIT License - feel free to use this for your own portfolio!
