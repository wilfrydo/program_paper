#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import time


# In[3]:


# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")


# In[5]:


# Tampilkan beberapa data pertama
df.head()


# In[7]:


# Deskripsi dataset
df.describe()


# In[9]:


# Cek informasi umum tentang dataset
df.info()


# In[10]:


# Cek missing value
print(df.isna().sum())


# In[11]:


# Hapus kolom id
df.drop('id', axis=1, inplace=True)


# In[12]:


# Tangani missing value pada bmi
df['bmi'].fillna(df['bmi'].median(), inplace=True)


# In[13]:


# Cek missing value
print(df.isna().sum())


# In[14]:


# Hapus duplikat
df.drop_duplicates(inplace=True)


# In[15]:


# Label encoding untuk kolom kategorikal
le = LabelEncoder()
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])


# In[16]:


# Buat fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']


# In[17]:


# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# In[26]:


print("Total data:", len(df))
print("Train set:", len(X_train))
print("Test set :", len(X_test))


# In[19]:


# Visualisasi distribusi target
sns.countplot(x = df['stroke'])
plt.title('Stroke Distribution')
plt.show()


# In[22]:


# ----------------------------------
# MODEL 1: Random Forest
# ----------------------------------
start_rf = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:,1]
end_rf = time.time()

# ----------------------------------
# MODEL 2: Naive Bayes
# ----------------------------------
start_nb = time.time()
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_probs = nb.predict_proba(X_test)[:,1]
end_nb = time.time()


# In[23]:


# ----------------------------------
# EVALUASI
# ----------------------------------
def evaluate_model(name, y_true, y_pred, y_probs, time_taken):
    print(f"\n=== {name} ===")
    print(f"Akurasi      : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision    : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score     : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Waktu (detik): {time_taken:.4f}")
    print(f"ROC-AUC      : {roc_auc_score(y_true, y_probs):.4f}")
    print(f"Avg Precision: {average_precision_score(y_true, y_probs):.4f}")

evaluate_model("Random Forest", y_test, rf_pred, rf_probs, end_rf - start_rf)
evaluate_model("Naive Bayes", y_test, nb_pred, nb_probs, end_nb - start_nb)


# In[24]:


# ----------------------------------
# CURVE VISUALIZATION
# ----------------------------------
def plot_curves(y_true, y_probs, name, color):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.3f}", color=color)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'ROC Curve - {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}', color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.show()

plot_curves(y_test, rf_probs, "Random Forest", "blue")
plot_curves(y_test, nb_probs, "Naive Bayes", "orange")


# In[25]:


# ----------------------------------
# THRESHOLDING (Optional: Mendemonstrasikan prediksi minoritas)
# ----------------------------------
print("\n=== Random Forest (Custom Threshold 0.2) ===")
thresh_rf = 0.2
rf_pred_thresh = (rf_probs > thresh_rf).astype(int)
print(f"Akurasi      : {accuracy_score(y_test, rf_pred_thresh):.4f}")
print(f"Precision    : {precision_score(y_test, rf_pred_thresh, zero_division=0):.4f}")
print(f"Recall       : {recall_score(y_test, rf_pred_thresh, zero_division=0):.4f}")
print(f"F1-Score     : {f1_score(y_test, rf_pred_thresh, zero_division=0):.4f}")


print("\n=== Naive Bayes (Custom Threshold 0.2) ===")
thresh_nb = 0.2
nb_pred_thresh = (nb_probs > thresh_nb).astype(int)
print(f"Akurasi      : {accuracy_score(y_test, nb_pred_thresh):.4f}")
print(f"Precision    : {precision_score(y_test, nb_pred_thresh, zero_division=0):.4f}")
print(f"Recall       : {recall_score(y_test, nb_pred_thresh, zero_division=0):.4f}")
print(f"F1-Score     : {f1_score(y_test, nb_pred_thresh, zero_division=0):.4f}")


# Catatan:
# - class_weight='balanced_subsample' pada RandomForest membantu agar model lebih memperhatikan kelas minoritas tanpa mengubah distribusi data.
# - Custom threshold menunjukkan bahwa dengan menurunkan threshold, recall terhadap kelas minoritas bisa naik (tapi precision mungkin turun).
# - Ini solusi tanpa oversampling/undersampling (tanpa membuat data seimbang secara eksplisit).


# In[ ]:

from sklearn.metrics import confusion_matrix

def show_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix ({title}):")
    print(cm)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Tampilkan confusion matrix untuk Random Forest
show_confusion_matrix(y_test, rf_pred, "Random Forest")

# Tampilkan confusion matrix untuk Naive Bayes
show_confusion_matrix(y_test, nb_pred, "Naive Bayes")


