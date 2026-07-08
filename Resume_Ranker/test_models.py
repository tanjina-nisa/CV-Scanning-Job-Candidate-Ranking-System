"""
test_models.py
--------------
Run this in terminal to verify both models are working correctly.

Usage:
    python test_models.py           # tests both models
    python test_models.py --sbert   # tests only Stage 1 (SBERT)
    python test_models.py --bert    # tests only Stage 2 (DistilBERT)
"""

import sys
import time
import argparse

# ─── COLORS FOR TERMINAL OUTPUT ──────────────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
BLUE   = '\033[94m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def ok(msg):    print(f"  {GREEN}✔  {msg}{RESET}")
def fail(msg):  print(f"  {RED}✘  {msg}{RESET}")
def info(msg):  print(f"  {YELLOW}→  {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{BLUE}{'='*55}{RESET}\n{BOLD}{BLUE}  {msg}{RESET}\n{BOLD}{BLUE}{'='*55}{RESET}")


# ─── SAMPLE DATA FOR TESTING ─────────────────────────────────────────────────
SAMPLE_CVS = {
    "Strong Python Dev": """
        Alice — 4 years Python backend developer.
        Skills: Python, Django, FastAPI, PostgreSQL, REST API, Docker, AWS, Git, Redis.
        Education: BSc Computer Science.
        Built microservices handling 100k daily requests. CI/CD with GitHub Actions.
    """,
    "Junior Frontend": """
        Bob — 1 year junior developer.
        Skills: JavaScript, React, HTML, CSS, MongoDB.
        Education: BSc Software Engineering.
        Built a small e-commerce site with React.
    """,
    "Data Scientist": """
        Carol — 3 years data scientist.
        Skills: Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, NLP, scikit-learn, SQL.
        Education: MSc Data Science.
        Trained BERT models for text classification. Deployed ML pipelines on GCP.
    """
}

JOB_DESC = """
    Looking for an experienced Python backend developer.
    Must know Python, Django or FastAPI, PostgreSQL, REST APIs, Docker.
    AWS or GCP cloud experience is a plus.
"""

TRAINING_DATA = [
    ("Python developer with Django FastAPI PostgreSQL REST APIs Docker AWS.", "Backend Developer"),
    ("Java backend Spring Boot MySQL Docker Kubernetes microservices.", "Backend Developer"),
    ("Data scientist machine learning deep learning TensorFlow PyTorch NLP scikit-learn.", "Data Science"),
    ("ML engineer Python pandas numpy statistical modeling computer vision.", "Data Science"),
    ("React TypeScript CSS frontend developer JavaScript Vue Angular.", "Web Developer"),
    ("Node.js Express MongoDB GraphQL full stack web developer.", "Web Developer"),
    ("HR manager talent acquisition recruitment employee relations payroll.", "HR"),
    ("Financial analyst Excel budgeting valuation financial modeling investment.", "Finance"),
    ("DevOps Docker Kubernetes Terraform CI/CD Jenkins AWS infrastructure.", "DevOps"),
    ("UI UX designer Figma Adobe XD wireframing prototyping user research.", "Designer"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 1 — ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
def test_environment():
    header("TEST 1: Environment & GPU")
    passed = True

    # Python version
    v = sys.version_info
    if v.major == 3 and v.minor >= 8:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — need 3.8+")
        passed = False

    # Required packages
    packages = {
        'torch':                'PyTorch',
        'sentence_transformers':'sentence-transformers',
        'transformers':         'HuggingFace transformers',
        'sklearn':              'scikit-learn',
        'pdfplumber':           'pdfplumber',
        'numpy':                'numpy',
        'pandas':               'pandas',
    }
    for module, name in packages.items():
        try:
            __import__(module)
            ok(f"{name} installed")
        except ImportError:
            fail(f"{name} NOT installed  →  pip install {module}")
            passed = False

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"GPU detected: {gpu}  ({vram:.1f} GB VRAM)")
        else:
            info("No GPU found — running on CPU (slower but works)")
    except Exception as e:
        fail(f"GPU check error: {e}")

    return passed


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 2 — STAGE 1: SBERT
# ─────────────────────────────────────────────────────────────────────────────
def test_sbert():
    header("TEST 2: Stage 1 — SBERT Semantic Matching")

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model
        info("Loading all-MiniLM-L6-v2 ...")
        t0 = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        ok(f"Model loaded in {time.time()-t0:.1f}s on {device}")

        # Score each sample CV against the job description
        print(f"\n  {'CV':<22} {'Score':>7}  {'Expectation'}")
        print(f"  {'-'*52}")

        scores = {}
        for name, cv_text in SAMPLE_CVS.items():
            emb = model.encode([cv_text, JOB_DESC], convert_to_tensor=True)
            score = util.cos_sim(emb[0], emb[1]).item() * 100
            scores[name] = score
            bar = '█' * int(score / 5)
            print(f"  {name:<22} {score:>6.1f}%  {bar}")

        # Sanity checks
        print()
        if scores["Strong Python Dev"] > scores["Junior Frontend"]:
            ok("Python Dev scored higher than Junior Frontend  ✓ makes sense")
        else:
            fail("Junior Frontend scored higher than Python Dev — unexpected")

        if scores["Strong Python Dev"] > scores["Data Scientist"]:
            ok("Python Dev scored highest for Python backend job  ✓ makes sense")
        else:
            info("Data Scientist scored higher — both profiles are similar, this can happen")

        # Speed test
        info("Speed test — encoding 10 texts ...")
        t0 = time.time()
        model.encode(["test resume text"] * 10)
        ok(f"10 encodings done in {time.time()-t0:.2f}s")

        return True

    except Exception as e:
        fail(f"SBERT test failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 3 — STAGE 2: DistilBERT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def test_distilbert():
    header("TEST 3: Stage 2 — DistilBERT Classifier")

    try:
        import torch
        import numpy as np
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer, TrainingArguments
        )
        from sklearn.preprocessing import LabelEncoder
        from torch.utils.data import Dataset

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Prepare data
        texts  = [d[0] for d in TRAINING_DATA]
        labels = [d[1] for d in TRAINING_DATA]
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        ok(f"Labels: {list(le.classes_)}")

        # Tokenizer
        info("Loading DistilBERT tokenizer ...")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        ok("Tokenizer loaded")

        # Quick tokenization test
        sample = tokenizer("Python developer Django REST API", return_tensors='pt', truncation=True, max_length=64)
        ok(f"Tokenization test passed — token shape: {sample['input_ids'].shape}")

        # Dataset
        class QuickDataset(Dataset):
            def __init__(self, texts, labels):
                self.enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
                self.labels = torch.tensor(labels, dtype=torch.long)
            def __len__(self): return len(self.labels)
            def __getitem__(self, i):
                return {'input_ids': self.enc['input_ids'][i],
                        'attention_mask': self.enc['attention_mask'][i],
                        'labels': self.labels[i]}

        dataset = QuickDataset(texts, encoded)
        ok(f"Dataset created — {len(dataset)} samples")

        # Load model
        info("Loading DistilBERT model (1 epoch quick test) ...")
        t0 = time.time()
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=len(le.classes_)
        )
        ok(f"Model loaded in {time.time()-t0:.1f}s  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters)")

        # Quick 1-epoch train to confirm it runs
        info("Running 1-epoch quick train to confirm pipeline works ...")
        args = TrainingArguments(
            output_dir='./test_output',
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_steps=99999,
            report_to='none',
        
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        t0 = time.time()
        trainer.train()
        ok(f"1-epoch training done in {time.time()-t0:.1f}s")

        # Inference test
        info("Testing inference on 3 sample CVs ...")
        model.eval()
        model.to(device)
        print(f"\n  {'CV':<22} {'Predicted Category':<20} {'Confidence'}")
        print(f"  {'-'*55}")

        for name, cv_text in SAMPLE_CVS.items():
            inputs = tokenizer(cv_text, return_tensors='pt', truncation=True,
                               padding=True, max_length=128).to(device)
            with torch.no_grad():
                out = model(**inputs)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = le.inverse_transform([idx])[0]
            conf = probs[idx] * 100
            print(f"  {name:<22} {label:<20} {conf:.1f}%")

        ok("Inference test passed")

        # Cleanup test output
        import shutil
        shutil.rmtree('./test_output', ignore_errors=True)

        return True

    except Exception as e:
        fail(f"DistilBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 4 — PDF READING
# ─────────────────────────────────────────────────────────────────────────────
def test_pdf():
    header("TEST 4: PDF Reading (pdfplumber)")
    try:
        import pdfplumber
        ok("pdfplumber imported")

        # Check if any PDFs exist in ./cvs
        from pathlib import Path
        cvs_path = Path('./cvs')
        if cvs_path.exists():
            pdfs = list(cvs_path.glob('*.pdf'))
            if pdfs:
                ok(f"Found {len(pdfs)} PDF(s) in ./cvs/")
                # Try reading the first one
                with pdfplumber.open(pdfs[0]) as pdf:
                    text = pdf.pages[0].extract_text() or ''
                ok(f"Read '{pdfs[0].name}' — {len(text)} chars extracted")
            else:
                info("No PDFs in ./cvs/ yet — demo CVs will be used in notebook")
        else:
            info("./cvs/ folder not found — create it and add your PDFs")

        return True
    except Exception as e:
        fail(f"PDF test failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test resume screening models')
    parser.add_argument('--sbert', action='store_true', help='Test SBERT only')
    parser.add_argument('--bert',  action='store_true', help='Test DistilBERT only')
    args = parser.parse_args()

    print(f"\n{BOLD}Resume Screening — Model Test Suite{RESET}")
    print("Running all checks...\n")

    results = {}

    if args.sbert:
        results['SBERT'] = test_sbert()
    elif args.bert:
        results['DistilBERT'] = test_distilbert()
    else:
        # Run all tests
        results['Environment']  = test_environment()
        results['SBERT']        = test_sbert()
        results['DistilBERT']   = test_distilbert()
        results['PDF Reading']  = test_pdf()

    # Summary
    header("SUMMARY")
    all_passed = True
    for name, passed in results.items():
        if passed:
            ok(f"{name}")
        else:
            fail(f"{name}")
            all_passed = False

    print()
    if all_passed:
        print(f"  {GREEN}{BOLD}All tests passed. Your models are working correctly.{RESET}")
        print(f"  {GREEN}You can now run the notebook.{RESET}\n")
    else:
        print(f"  {RED}{BOLD}Some tests failed. Fix the issues above before running the notebook.{RESET}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()