An end-to-end AI-powered recruitment automation tool that ranks resumes against job requirements using a hybrid of Machine Learning and Deep Learning models. Built to reduce the time and bias involved in manual resume screening.
Key Features:

📄 PDF resume parsing with automated skill, experience, and education extraction
🤖 4-model pipeline: Logistic Regression & Random Forest (TF-IDF classification), DistilBERT (fine-tuned contextual classification), SBERT (semantic similarity matching)
⚖️ Hybrid scoring formula: Semantic Similarity (40%) + Skill Match (30%) + Experience (15%) + Education (15%)
🔍 Explainable AI output — matched/missing skills, score breakdowns, and improvement suggestions
📊 Streamlit dashboard with HR login, batch upload, candidate ranking, percentile analytics, and CSV export
✅ 99%+ accuracy and F1 scores across all classification models on a 962-sample resume dataset (25 job categories)

Tech Stack: Python 3.12 · PyTorch · HuggingFace Transformers · scikit-learn · Sentence-Transformers · Streamlit · pdfplumber · pandas
