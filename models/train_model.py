import pickle, json, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import os

STOP_WORDS = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','this','that','these','those','it','its','i','me','my','we','our','you','your','he','him','his','she','her','they','them','their','what','which','who','when','where','why','how','all','each','also','said','says','according','told','report','news','today'}

def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

REAL = [
    "Prime Minister Modi inaugurated the new metro line in Bangalore calling it a milestone for urban mobility in Karnataka.",
    "Reserve Bank of India kept interest rates unchanged at 6.5 percent amid concerns about inflation and global economic uncertainty.",
    "India GDP grew at 7.2 percent in Q3 making it the fastest growing major economy according Ministry Statistics.",
    "ISRO successfully launched PSLV-C57 mission carrying Aditya-L1 solar observation spacecraft from Sriharikota.",
    "Supreme Court of India upheld right to privacy as a fundamental right in landmark judgment affecting millions citizens.",
    "Tata Motors reported record EV sales in October with Nexon EV leading segment across major Indian cities.",
    "Indian cricket team defeated Australia in Test series with Virat Kohli scoring twin centuries final match.",
    "Heavy rainfall caused flooding in Chennai with Tamil Nadu government deploying NDRF teams for rescue operations.",
    "Union Budget allocated Rs 11 lakh crore for infrastructure development highest ever capital expenditure announced.",
    "Scientists at IIT Bombay developed biodegradable plastic alternative using agricultural waste from sugarcane farms.",
    "India export pharmaceuticals crossed USD 25 billion cementing position as pharmacy world international trade.",
    "Election Commission announced schedule for assembly elections five states to be held November December.",
    "Reliance Industries announced Rs 75000 crore investment in green energy projects across Gujarat Rajasthan.",
    "Government launched PM Vishwakarma scheme providing training credit support 18 traditional crafts categories.",
    "India signed free trade agreement UAE covering tariff reductions over 90 percent goods bilateral trade.",
    "Indian Railways completed electrification 95 percent broad gauge network achieving energy efficiency goals.",
    "Air India announced order 470 aircraft from Airbus Boeing biggest aviation deal Indian history contract.",
    "Jan Dhan scheme achieved 50 crore account milestone total deposits exceeding Rs 2 lakh crore nationwide.",
    "India digital payment ecosystem processed transactions worth Rs 1800 lakh crore through UPI methods.",
    "Chandrayaan-3 data revealed presence sulphur several elements near lunar south pole confirmed scientists.",
    "India startup ecosystem reached 100 unicorn milestone several emerging tier 2 tier 3 cities success.",
    "Government launched PLI scheme semiconductors attracting investments global chipmakers including Micron factory.",
    "Food inflation India declined 4.6 percent August driven fall vegetable prices markets nationwide.",
    "Indian Navy commissioned INS Vindhyagiri stealth frigate built Garden Reach Shipbuilders Kolkata yard.",
    "GST revenue collection crossed Rs 2 lakh crore first time April indicating strong economic growth.",
    "Vande Bharat Express network expanded connect 50 cities improving rail connectivity across country.",
    "Ayushman Bharat scheme expanded coverage senior citizens above 70 years providing free health insurance.",
    "National Education Policy implementation showed significant increase gross enrollment ratios across states.",
    "India signed defence cooperation agreement with France for advanced submarine technology transfer project.",
    "Manipur government formed peace committees ease tensions between communities after months ethnic conflict resolution.",
]

FAKE = [
    "SHOCKING: Government secretly planning replace all Indian currency digital tokens midnight tonight warning!",
    "EXPOSED: NASA confirms alien base discovered moon directly above India government hiding truth decades!",
    "BREAKING: Scientists confirm drinking cow urine daily cures cancer diabetes all major diseases pharma suppressing!",
    "VIRAL: Modi government planning impose martial law 5 states before elections army deployed secretly tonight!",
    "WhatsApp FORWARD: Eating garlic turmeric 3 AM for 7 days removes all COVID variants permanently body!",
    "SHOCKING: Supreme Court secretly ruled Pakistan given Kashmir 2025 media hiding from Indians forward!",
    "EXPOSED: RBI printing unlimited money give free Rs 15 lakh every Indian citizen next month apply!",
    "BREAKING: China troops entered 50 km inside Indian territory government hiding public citizens beware!",
    "VIRAL TRUTH: 5G towers spreading new virus targeting Hindus specifically government ignores despite proof!",
    "SHOCKING: All petrol become free India January 1 government subsidize 100 percent fuel cost forward!",
    "EXPOSED: Bill Gates plans microchip all Indians upcoming vaccine drive chips track movements everywhere!",
    "BREAKING: India become part China 2026 due secret treaty signed previous government hidden agenda!",
    "SHOCKING: New law allow government seize property anyone posting against ruling party social media!",
    "EXPOSED: COVID-19 created Indian lab released globally who covering truth government hidden conspiracy!",
    "BREAKING: Petrol prices rise Rs 500 litre this week India runs out oil reserves completely tonight!",
    "VIRAL FORWARD: Eating 5 bananas 3 onions daily makes corona antibodies 1000 times stronger vaccine!",
    "SHOCKING: Government plans ban WhatsApp all social media India permanently next month spread awareness!",
    "VIRAL: New government scheme gives Rs 5 lakh every family submits Aadhaar number WhatsApp tonight!",
    "SHOCKING: Mysterious planet approaching earth cause massive earthquake India December 21 evacuate!",
    "EXPOSED: Rohingya given Indian citizenship secretly will vote next election change results forever!",
    "BREAKING: New disease worse COVID spreading China infected 10 lakh people India secretly hiding!",
    "SHOCKING: Election EVMs hacked simple TV remote opposition parties rigged all state elections proof!",
    "EXPOSED: Delhi government secretly poisoning tap water control population growth foreign power orders!",
    "BREAKING: India Pakistan war starting tomorrow government secret alert military evacuate border states!",
    "VIRAL: Coca Cola contains ingredient makes Hindus convert Christianity FDA data foreign conspiracy!",
    "SHOCKING: Cancer medicine Rs 100 suppressed pharma companies pay doctors Rs 10 lakh hide it!",
    "EXPOSED: America planning nuclear bomb Pakistan all India destroyed radiation tomorrow morning!",
    "VIRAL TRUTH: Pizza burger causes genetic mutation turns children alternative lifestyle Western conspiracy!",
    "SHOCKING: New satellite shows Pakistan secretly building nuclear missiles aimed Delhi Mumbai cities!",
    "BREAKING URGENT: Government secretly sold Andaman Nicobar Islands China deal signed midnight!",
]

real_df = pd.DataFrame({'text': REAL, 'label': 1})
fake_df = pd.DataFrame({'text': FAKE, 'label': 0})
df = pd.concat([real_df]*4 + [fake_df]*4, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['processed'] = df['text'].apply(preprocess)

X_tr, X_te, y_tr, y_te = train_test_split(df['processed'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,3), sublinear_tf=True, min_df=1)),
    ('clf', LogisticRegression(C=5.0, max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42))
])
pipe.fit(X_tr, y_tr)

y_pred = pipe.predict(X_te)
y_prob = pipe.predict_proba(X_te)[:,1]
acc = accuracy_score(y_te, y_pred)
auc = roc_auc_score(y_te, y_prob)
cm = confusion_matrix(y_te, y_pred)
print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
print(f"CM:\n{cm}")

os.makedirs('models', exist_ok=True)
metrics = {
    "accuracy": round(acc*100, 2),
    "roc_auc": round(auc*100, 2),
    "precision_fake": round(float(cm[0,0])/max(float(cm[0,0]+cm[1,0]),1)*100, 2),
    "recall_fake": round(float(cm[0,0])/max(float(cm[0,0]+cm[0,1]),1)*100, 2),
    "precision_real": round(float(cm[1,1])/max(float(cm[1,1]+cm[0,1]),1)*100, 2),
    "recall_real": round(float(cm[1,1])/max(float(cm[1,1]+cm[1,0]),1)*100, 2),
    "confusion_matrix": cm.tolist(),
    "total_samples": len(df),
    "train_samples": len(X_tr),
    "test_samples": len(X_te),
}
json.dump(metrics, open('models/metrics.json','w'), indent=2)
pickle.dump(pipe, open('models/model.pkl','wb'))
print("Saved! ✓")
