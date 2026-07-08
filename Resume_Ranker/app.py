import streamlit as st
import pdfplumber
import torch
import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #c8f5e4 0%, #b8d9f5 50%, #c5c0f0 100%); min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1rem !important; max-width: 720px !important; margin: 0 auto !important; }
.navbar { display:flex; justify-content:space-between; align-items:center; padding:1rem 1.5rem; background:rgba(255,255,255,0.6); backdrop-filter:blur(12px); border-radius:14px; border:1px solid rgba(255,255,255,0.8); margin-bottom:2rem; }
.navbar-brand { font-size:1.1rem; font-weight:700; color:#4f46e5; }
.navbar-user { font-size:0.82rem; color:#6b7280; }
.card { background:rgba(255,255,255,0.6); backdrop-filter:blur(12px); border-radius:16px; border:1px solid rgba(255,255,255,0.8); padding:1.8rem; margin-bottom:1.2rem; }
.page-title { font-size:1.5rem; font-weight:700; color:#1f1f2e; margin-bottom:0.3rem; }
.page-sub { font-size:0.88rem; color:#6b7280; margin-bottom:1.5rem; }
.label { font-size:0.78rem; font-weight:600; color:#4f46e5; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background:rgba(255,255,255,0.75) !important; border:1.5px solid rgba(99,102,241,0.25) !important; border-radius:10px !important; font-family:'Inter',sans-serif !important; font-size:0.9rem !important; color:#1f1f2e !important; padding:0.6rem 0.9rem !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color:#4f46e5 !important; box-shadow:0 0 0 3px rgba(79,70,229,0.1) !important; }
.stTextInput label, .stTextArea label, .stNumberInput label, .stSelectbox label { font-size:0.85rem !important; font-weight:500 !important; color:#374151 !important; }
.stSelectbox > div > div { background:rgba(255,255,255,0.75) !important; border:1.5px solid rgba(99,102,241,0.25) !important; border-radius:10px !important; }
.stNumberInput > div > div > input { background:rgba(255,255,255,0.75) !important; border:1.5px solid rgba(99,102,241,0.25) !important; border-radius:10px !important; font-family:'Inter',sans-serif !important; }
.stButton > button { background:#4f46e5 !important; color:white !important; border:none !important; border-radius:10px !important; font-family:'Inter',sans-serif !important; font-weight:600 !important; font-size:0.9rem !important; padding:0.6rem 1.5rem !important; transition:all 0.2s !important; }
.stButton > button:hover { background:#4338ca !important; transform:translateY(-1px) !important; box-shadow:0 4px 12px rgba(79,70,229,0.3) !important; }
.stFileUploader > div { background:rgba(255,255,255,0.5) !important; border:2px dashed rgba(79,70,229,0.35) !important; border-radius:12px !important; }
.stFileUploader label { font-size:0.85rem !important; font-weight:500 !important; color:#374151 !important; }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin:1rem 0; }
.metric-item { background:rgba(255,255,255,0.65); border-radius:12px; padding:0.9rem; text-align:center; border:1px solid rgba(255,255,255,0.9); }
.metric-val { font-size:1.3rem; font-weight:700; color:#4f46e5; }
.metric-lbl { font-size:0.72rem; color:#6b7280; margin-top:0.15rem; font-weight:500; }
.badge { display:inline-block; padding:0.25rem 0.9rem; border-radius:6px; font-size:0.82rem; font-weight:600; }
.badge-green { background:#d1fae5; color:#065f46; }
.badge-yellow { background:#fef3c7; color:#92400e; }
.badge-red { background:#fee2e2; color:#991b1b; }
.badge-purple { background:#ede9fe; color:#4f46e5; }
.badge-blue { background:#dbeafe; color:#1e40af; }
.badge-pink { background:#fce7f3; color:#9d174d; }
.pill-green { display:inline-block; background:#d1fae5; color:#065f46; border-radius:6px; padding:0.18rem 0.6rem; margin:0.12rem; font-size:0.78rem; font-weight:500; }
.pill-red { display:inline-block; background:#fee2e2; color:#991b1b; border-radius:6px; padding:0.18rem 0.6rem; margin:0.12rem; font-size:0.78rem; font-weight:500; }
.box-insight { background:rgba(79,70,229,0.07); border-left:3px solid #4f46e5; border-radius:0 8px 8px 0; padding:0.75rem 1rem; font-size:0.85rem; color:#2d2d4e; margin:0.6rem 0; line-height:1.6; }
.box-tip { background:rgba(16,185,129,0.07); border-left:3px solid #10b981; border-radius:0 8px 8px 0; padding:0.75rem 1rem; font-size:0.85rem; color:#064e3b; margin:0.6rem 0; line-height:1.6; }
.divider { height:1px; background:rgba(79,70,229,0.15); margin:1.2rem 0; }
.stDownloadButton > button { background:rgba(255,255,255,0.7) !important; color:#4f46e5 !important; border:1.5px solid #4f46e5 !important; border-radius:10px !important; font-family:'Inter',sans-serif !important; font-weight:600 !important; }
.stDownloadButton > button:hover { background:#4f46e5 !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

USERS = {"hr@company.com": "hr1234", "admin@company.com": "admin123"}
for k, v in [('logged_in',False),('page','scan'),('results',None),('user','')]:
    if k not in st.session_state: st.session_state[k] = v

# LOGIN
if not st.session_state['logged_in']:
    st.markdown('<div style="text-align:center;padding:3rem 0 2rem;"><div style="font-size:2rem;font-weight:700;color:#4f46e5;">Resume Screener</div><div style="font-size:0.92rem;color:#6b7280;margin-top:0.4rem;">Smart CV matching powered by ML &amp; DL</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    email    = st.text_input("Email", placeholder="hr@company.com")
    password = st.text_input("Password", type="password", placeholder="Enter password")
    if st.button("Log In", use_container_width=True):
        if email in USERS and USERS[email] == password:
            st.session_state.update({'logged_in':True,'user':email})
            st.rerun()
        else:
            st.error("Wrong email or password.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# MODELS
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert  = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    clf    = DistilBertForSequenceClassification.from_pretrained('./saved_distilbert_classifier')
    tok    = DistilBertTokenizerFast.from_pretrained('./saved_distilbert_classifier')
    with open('./saved_distilbert_classifier/label_classes.json') as f: classes = json.load(f)
    le = LabelEncoder(); le.classes_ = np.array(classes)
    return sbert, clf, tok, le, device

sbert_model, clf_model, tokenizer, label_enc, device = load_models()

SKILLS = ['html','css','figma','adobe xd',
'oop','object oriented','data structures','algorithms','design patterns',
'solid','microservices','agile','scrum','kanban','tdd','bdd',
'junit','pytest','selenium','postman','jira','confluence',
'spring','hibernate','maven','gradle','npm','yarn',
'linux','bash','shell','powershell','vim',
'c','c++','c#','php','ruby','swift','kotlin','scala','rust',
'excel','power bi','looker','qlik','sas','spss','matlab',
'opencv','nltk','spacy','gensim','fasttext',
'redis','rabbitmq','activemq','grpc','graphql',
'terraform','ansible','puppet','chef','vagrant',
'firebase','supabase','amplify','heroku','netlify','vercel',
'recruitment','employee management','ms office','communication',
'onboarding','payroll','performance management','talent acquisition',
'employee relations','training and development','hris','workday','bamboohr',
'conflict resolution','labor law','compensation','benefits administration',
'sourcing','interviewing','background check','offer management',
'sap hr','oracle hcm','succession planning','workforce planning',
'sales','negotiation','customer service','account management',
'business development','crm','salesforce','hubspot','lead generation',
'project management','stakeholder management','risk management','budgeting',
'ms excel','ms word','ms powerpoint','google sheets','google docs',
'presentation','documentation','report writing','data entry',
'autocad','solidworks','catia','matlab','ansys','staad pro',
'circuit design','plc','scada','embedded systems','microcontroller',
'accounting','bookkeeping','financial analysis','tally','quickbooks',
'tax','audit','financial modeling','valuation','investment',
'content writing','seo','social media','digital marketing','google ads',
'facebook ads','email marketing','wordpress','shopify']
EDU_L = {'phd':4,'ph.d':4,'doctorate':4,'master':3,'msc':3,'mba':3,'bachelor':2,'bsc':2,'b.sc':2,'btech':2,'b.tech':2,'diploma':1,'hsc':1,'ssc':0}
EDU_R = {'any':0,'diploma':1,'bachelor':2,'master':3,'phd':4}

def read_pdf(f):
    t = ''
    try:
        with pdfplumber.open(f) as p:
            for pg in p.pages:
                tx = pg.extract_text()
                if tx: t += tx + '\n'
    except: pass
    # Normalize — add space after comma, clean whitespace
    t = re.sub(r',', ', ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def get_skills(text):
    tl = text.lower()
    tl = re.sub(r'[,;/|]', ' ', tl)
    tl_clean = re.sub(r'[^a-z0-9\s]', ' ', tl)
    words = set(tl_clean.split())
    found = []
    for s in SKILLS:
        s_clean = re.sub(r'[^a-z0-9\s]', ' ', s).strip()
        s_words = s_clean.split()
        # Single short keyword — whole word match only
        if len(s_words) == 1 and len(s_clean) <= 4:
            if s_clean in words:
                found.append(s)
        # Multi-word or longer keyword — all words must appear
        elif all(w in tl_clean for w in s_words):
            found.append(s)
        # Compound match without spaces/hyphens
        elif len(s) > 5 and s.replace(' ','').replace('-','') in tl.replace(' ','').replace('-','').replace('_',''):
            found.append(s)
    return list(set(found))
def get_exp(text):
    pats=[r'(\d+\.?\d*)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',r'(\d+\.?\d*)\+?\s*years?\s*(?:backend|frontend|python|data|ml|software|total)']
    yrs=[]
    for p in pats: yrs+=[float(m) for m in re.findall(p,text.lower())]
    return max(yrs) if yrs else 0.0
def get_edu(text):
    tl,best=text.lower(),('none',0)
    for kw,lv in EDU_L.items():
        if kw in tl and lv>best[1]: best=(kw,lv)
    return best
def predict_cat(text):
    inp=tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=256).to(device)
    clf_model.to(device); clf_model.eval()
    with torch.no_grad(): out=clf_model(**inp)
    probs=torch.softmax(out.logits,dim=1).cpu().numpy()[0]; idx=int(np.argmax(probs))
    return label_enc.inverse_transform([idx])[0],round(float(probs[idx])*100,1)

def score_cv(text,skills,req_skills,job_desc,min_exp,cv_exp,req_edu,cv_edu_lv):
    emb=sbert_model.encode([text,job_desc],convert_to_tensor=True)
    sbert=max(0.0,util.cos_sim(emb[0],emb[1]).item())
    
    req=[s.strip().lower() for s in req_skills if s.strip()]
    
    # Search directly in CV text — not just SKILLS list
    cv_text_lower = re.sub(r'[,;/|]', ' ', text.lower())
    cv_text_clean = re.sub(r'[^a-z0-9\s]', ' ', cv_text_lower)
    cv_words = set(cv_text_clean.split())
    
    def skill_in_cv(skill):
        s_clean = re.sub(r'[^a-z0-9\s]', ' ', skill).strip()
        s_words = s_clean.split()
        if len(s_words)==1 and len(s_clean)<=4:
            return s_clean in cv_words
        elif all(w in cv_text_clean for w in s_words):
            return True
        elif len(skill)>5 and skill.replace(' ','').replace('-','') in cv_text_lower.replace(' ','').replace('-',''):
            return True
        return False
    
    matched=[s for s in req if skill_in_cv(s)]
    missing=[s for s in req if not skill_in_cv(s)]
    
    sk_s=len(matched)/len(req) if req else 1.0
    ex_s=min(cv_exp/min_exp,1.0) if min_exp>0 else 1.0
    rl=EDU_R.get(req_edu.lower(),0); ed_s=1.0 if cv_edu_lv>=rl else (0.5 if cv_edu_lv==rl-1 else 0.0)
    final=0.40*sbert+0.30*sk_s+0.15*ex_s+0.15*ed_s
    return dict(final=round(final*100,1),sbert=round(sbert*100,1),skill=round(sk_s*100,1),exp=round(ex_s*100,1),edu=round(ed_s*100,1),matched=matched,missing=missing)

def verdict(s):
    if s>=75: return "Highly Recommended","badge-green"
    if s>=55: return "Moderate Match","badge-yellow"
    return "Not Recommended","badge-red"

def pct_label(s,all_s):
    p10,p90=np.percentile(all_s,10),np.percentile(all_s,90)
    if s>=p90: return "Top 10%","badge-purple"
    if s>=p10: return "Top 50%","badge-blue"
    return "Bottom 50%","badge-pink"

def explain(r):
    parts=[]
    if r['sbert']>=65: parts.append("Strong semantic alignment with job description.")
    elif r['sbert']>=45: parts.append("Moderate semantic match with job description.")
    else: parts.append("Low semantic similarity — CV language differs from job description.")
    if r['skill']>=80: parts.append(f"Excellent skill coverage ({len(r['matched'])} matched).")
    elif r['skill']>=50: parts.append(f"Partial skill match — {len(r['matched'])} matched, {len(r['missing'])} missing.")
    else: parts.append(f"Low skill match — only {len(r['matched'])} required skills found.")
    if r['exp']==100: parts.append("Meets experience requirement.")
    elif r['exp']>=50: parts.append("Partially meets experience requirement.")
    else: parts.append("Below required experience level.")
    return " ".join(parts)

def tips(r):
    t=[]
    if r['missing']: t.append(f"Develop missing skills: {', '.join(r['missing'][:4])}")
    if r['exp']<100: t.append("Gain more relevant work experience")
    if r['sbert']<50: t.append("Tailor CV language to match job description")
    return t or ["Strong profile — no major gaps"]

# NAVBAR
st.markdown(f'<div class="navbar"><span class="navbar-brand">📄 Resume Screener</span><span class="navbar-user">👤 {st.session_state["user"]}</span></div>', unsafe_allow_html=True)
col_x,col_logout=st.columns([6,1])
with col_logout:
    if st.button("Logout"): st.session_state.update({'logged_in':False,'results':None,'page':'scan'}); st.rerun()

# ═══════ PAGE: SCAN ═══════
if st.session_state['page']=='scan' or not st.session_state['results']:
    st.markdown('<div class="page-title">Screen Candidates</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload CVs and set job requirements to get AI-powered rankings.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Upload CVs (PDF)</div>', unsafe_allow_html=True)
    uploaded=st.file_uploader("",type=["pdf"],accept_multiple_files=True,label_visibility="collapsed")
    if uploaded:
        for f in uploaded: st.markdown(f"<span style='font-size:0.82rem;color:#4f46e5;'>✔ {f.name}</span>",unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Job Requirements</div>', unsafe_allow_html=True)
    position=st.text_input("Position",placeholder="e.g. Python Backend Developer")
    requirement=st.text_input("Required Skills",placeholder="e.g. python, django, postgresql, docker")
    c1,c2=st.columns(2)
    with c1: min_exp=st.number_input("Min. Experience (years)",0.0,20.0,1.0,0.5)
    with c2: req_edu=st.selectbox("Education Level",["any","diploma","bachelor","master","phd"])
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Scan Candidates",use_container_width=True):
        if not position: st.warning("Enter a job position.")
        elif not requirement: st.warning("Enter required skills.")
        elif not uploaded: st.warning("Upload at least one CV.")
        else:
            req_skills=[s.strip().lower() for s in requirement.split(',')]
            job_desc=f"Looking for a {position} with expertise in {requirement}. Strong knowledge of {requirement} required."
            results=[]
            with st.spinner("Analyzing CVs..."):
                for f in uploaded:
                    text=read_pdf(f); skills=get_skills(text); exp=get_exp(text)
                    edu_l,edu_lv=get_edu(text); cat,conf=predict_cat(text)
                    sc=score_cv(text,skills,req_skills,job_desc,min_exp,exp,req_edu,edu_lv)
                    results.append({'name':f.name,'exp':exp,'edu_l':edu_l,'category':cat,'conf':conf,**sc})
            results=sorted(results,key=lambda x:x['final'],reverse=True)
            st.session_state.update({'results':results,'position':position,'req_skills':req_skills,'page':'output'})
            st.rerun()

# ═══════ PAGE: OUTPUT ═══════
elif st.session_state['page']=='output':
    results=st.session_state['results']
    position=st.session_state.get('position','')
    all_scores=[r['final'] for r in results]
    MEDALS={1:'🥇',2:'🥈',3:'🥉'}

    if st.button("← Back"): st.session_state['page']='scan'; st.rerun()

    st.markdown(f'<div class="page-title">{position}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{len(results)} candidate(s) analyzed</div>',unsafe_allow_html=True)

    # Chart
    if len(results)>1:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,3.5))
        names=[r['name'].replace('.pdf','')[:18] for r in results]
        scores=[r['final'] for r in results]
        clrs=['#4f46e5' if i==0 else '#818cf8' if i==1 else '#a5b4fc' if i==2 else '#c7d2fe' for i in range(len(results))]
        bars=ax.bar(names,scores,color=clrs,edgecolor='white',linewidth=1.2,width=0.55)
        for bar,s in zip(bars,scores):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{s}%',ha='center',va='bottom',fontsize=9,fontweight='600',color='#1f1f2e')
        ax.set_ylim(0,110); ax.set_ylabel('Match Score (%)',fontsize=9,color='#6b7280')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb'); ax.spines['bottom'].set_color('#e5e7eb')
        ax.tick_params(colors='#6b7280',labelsize=8); ax.set_facecolor('none'); fig.patch.set_alpha(0)
        plt.xticks(rotation=12,ha='right'); plt.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    # Candidates
    st.markdown('<div class="label" style="margin-bottom:0.8rem;">Candidate Results</div>',unsafe_allow_html=True)
    for i,r in enumerate(results,1):
        medal=MEDALS.get(i,f'#{i}'); vrd,vc=verdict(r['final']); pl,pc=pct_label(r['final'],all_scores)
        with st.expander(f"{medal}  {r['name']}  ·  {r['final']}%",expanded=(i==1)):
            st.markdown(f'<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem;align-items:center;"><span class="badge {vc}">{vrd}</span><span class="badge {pc}">{pl}</span><span class="badge badge-blue">{r["category"]} · {r["conf"]}%</span><span style="font-size:0.8rem;color:#6b7280;margin-left:auto;">Exp: {r["exp"]}y · Edu: {r["edu_l"]}</span></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-grid"><div class="metric-item"><div class="metric-val">{r["final"]}%</div><div class="metric-lbl">Overall</div></div><div class="metric-item"><div class="metric-val">{r["sbert"]}%</div><div class="metric-lbl">Semantic</div></div><div class="metric-item"><div class="metric-val">{r["skill"]}%</div><div class="metric-lbl">Skills</div></div><div class="metric-item"><div class="metric-val">{r["exp"]}%</div><div class="metric-lbl">Experience</div></div></div><div class="divider"></div>',unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**✅ Matched Skills**")
                st.markdown(''.join([f'<span class="pill-green">{s}</span>' for s in r['matched']]) or "<span style='font-size:0.82rem;color:#6b7280'>None detected</span>",unsafe_allow_html=True)
            with c2:
                st.markdown("**❌ Missing Skills**")
                st.markdown(''.join([f'<span class="pill-red">{s}</span>' for s in r['missing']]) or "<span style='font-size:0.82rem;color:#065f46'>All matched!</span>",unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="box-insight">🔍 <b>Score Explanation</b><br>{explain(r)}</div>',unsafe_allow_html=True)
            tip_html=''.join([f'<li style="margin:0.2rem 0;">{t}</li>' for t in tips(r)])
            st.markdown(f'<div class="box-tip">💡 <b>Improvement Suggestions</b><ul style="margin:0.3rem 0 0 1.2rem;padding:0;">{tip_html}</ul></div>',unsafe_allow_html=True)

    # Analytics
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-bottom:0.8rem;">Advanced Analytics</div>',unsafe_allow_html=True)
    p10=np.percentile(all_scores,10); p90=np.percentile(all_scores,90)
    top=[r for r in results if r['final']>=p90]; mid=[r for r in results if p10<=r['final']<p90]; bot=[r for r in results if r['final']<p10]
    ca,cb,cc=st.columns(3)
    with ca: st.markdown(f'<div class="card" style="text-align:center;border-top:3px solid #4f46e5;"><div style="font-size:0.78rem;font-weight:600;color:#4f46e5;">TOP 10%</div><div style="font-size:1.8rem;font-weight:700;color:#1f1f2e;">{len(top)}</div><div style="font-size:0.78rem;color:#6b7280;">≥ {p90:.0f}%</div></div>',unsafe_allow_html=True)
    with cb: st.markdown(f'<div class="card" style="text-align:center;border-top:3px solid #60a5fa;"><div style="font-size:0.78rem;font-weight:600;color:#1d4ed8;">MIDDLE 50%</div><div style="font-size:1.8rem;font-weight:700;color:#1f1f2e;">{len(mid)}</div><div style="font-size:0.78rem;color:#6b7280;">{p10:.0f}%–{p90:.0f}%</div></div>',unsafe_allow_html=True)
    with cc: st.markdown(f'<div class="card" style="text-align:center;border-top:3px solid #f472b6;"><div style="font-size:0.78rem;font-weight:600;color:#9d174d;">BOTTOM 10%</div><div style="font-size:1.8rem;font-weight:700;color:#1f1f2e;">{len(bot)}</div><div style="font-size:0.78rem;color:#6b7280;">< {p10:.0f}%</div></div>',unsafe_allow_html=True)

    g1,g2=st.columns(2)
    with g1:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown("**🏆 Top Candidates — Strengths**")
        top_pool=top if top else results[:max(1,len(results)//3)]
        top_sk={}
        for r in top_pool:
            for s in r['matched']: top_sk[s]=top_sk.get(s,0)+1
        if top_sk:
            common=sorted(top_sk.items(),key=lambda x:-x[1])[:8]
            st.markdown(''.join([f'<span class="pill-green">{s}</span>' for s,_ in common]),unsafe_allow_html=True)
        avg_t=np.mean([r['final'] for r in top_pool])
        st.markdown(f'<div class="box-insight" style="margin-top:0.8rem;">Average score: <b>{avg_t:.1f}%</b><br>Top candidates show strong skill coverage and semantic alignment.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown("**⚠️ Lower Candidates — Weaknesses**")
        bot_pool=bot if bot else results[-(max(1,len(results)//3)):]
        bot_sk={}
        for r in bot_pool:
            for s in r['missing']: bot_sk[s]=bot_sk.get(s,0)+1
        if bot_sk:
            common2=sorted(bot_sk.items(),key=lambda x:-x[1])[:8]
            st.markdown(''.join([f'<span class="pill-red">{s}</span>' for s,_ in common2]),unsafe_allow_html=True)
        avg_b=np.mean([r['final'] for r in bot_pool])
        st.markdown(f'<div class="box-tip" style="margin-top:0.8rem;">Average score: <b>{avg_b:.1f}%</b><br>Lower candidates lack key skills and semantic alignment.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    if len(results)>=2:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown("**Score Breakdown — Top Candidates**")
        top5=results[:min(5,len(results))]; cats=['Semantic','Skills','Experience','Education']
        clr2=['#4f46e5','#60a5fa','#a78bfa','#34d399','#f472b6']; x=np.arange(len(cats)); w=0.7/len(top5)
        fig2,ax2=plt.subplots(figsize=(8,3.8))
        for idx,r in enumerate(top5):
            vals=[r['sbert'],r['skill'],r['exp'],r['edu']]; offset=(idx-len(top5)/2+0.5)*w
            ax2.bar(x+offset,vals,w*0.9,label=r['name'].replace('.pdf','')[:14],color=clr2[idx],alpha=0.88,edgecolor='white')
        ax2.set_xticks(x); ax2.set_xticklabels(cats,fontsize=9); ax2.set_ylim(0,115)
        ax2.set_ylabel('Score (%)',fontsize=9,color='#6b7280'); ax2.legend(fontsize=8,loc='upper right',framealpha=0.5)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#e5e7eb'); ax2.spines['bottom'].set_color('#e5e7eb')
        ax2.tick_params(colors='#6b7280',labelsize=8); ax2.set_facecolor('none'); fig2.patch.set_alpha(0)
        plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    df_out=pd.DataFrame([{'Rank':i+1,'CV':r['name'],'Score (%)':r['final'],'Semantic (%)':r['sbert'],'Skills (%)':r['skill'],'Experience (%)':r['exp'],'Education (%)':r['edu'],'Matched':', '.join(r['matched']),'Missing':', '.join(r['missing']),'Category':r['category'],'Verdict':verdict(r['final'])[0],'Percentile':pct_label(r['final'],all_scores)[0]} for i,r in enumerate(results)])
    st.download_button("⬇️ Download Report (CSV)",data=df_out.to_csv(index=False).encode('utf-8'),file_name=f"screening_{position.replace(' ','_')}.csv",mime='text/csv',use_container_width=True)
