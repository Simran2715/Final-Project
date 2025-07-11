# For pass or fail project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, auc, roc_curve,roc_auc_score

student_performance=pd.read_csv('Student_Performance.csv')

X=student_performance.drop(columns=['pass_fail','future_score','topic_difficulty'])
y=student_performance['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

le=LabelEncoder()
y_encode=le.fit_transform(y)

scaler= StandardScaler()
X_train_scale= scaler.fit_transform(X_train)
X_test_scale=scaler.fit_transform(X_test)

lr=LogisticRegression(random_state=42)
lr.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred_lr=lr.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred_lr)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred_lr))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred_lr)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression Model')
plt.show()

feature_importance= pd.DataFrame({
    'Feature' : X.columns,
    'Coefficient' : lr.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print('\n-----Important Features-----')
print(feature_importance)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='viridis')
plt.title("Feature Importance for predicting Pass/Fail")
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--',alpha=0.6)
plt.show()

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred_rf=rf.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred_rf)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred_rf))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred_rf)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier Model')
plt.show()

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred_rf=rf.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred_rf)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred_rf))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred_rf)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier Model')
plt.show()

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred=knn.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KN Classifier Model')
plt.show()

mlp=MLPClassifier(random_state=42)
mlp.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred=mlp.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix MLP Classifier')

svm=SVC(random_state=42)
svm.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred=svm.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVC')

tree=DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred=tree.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

bnb=BernoulliNB()
bnb.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred=bnb.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=42)
gbc.fit(X_train_scale, y_train)
print('\n Model training complete')
y_pred=gbc.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy: .4f}')
print('\n Classification Report :')
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix :")
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True ,fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

import pickle

with open ('rf.pkl','wb') as file:
    pickle.dump(rf, file)

# for score range project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_curve,auc, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

label_encode={}
categorical_col=['pass_fail','topic_difficulty']

for col in categorical_col:
    label=LabelEncoder()
    student_performance[col]=label.fit_transform(student_performance[col])
    label_encode[col]=label

X=student_performance.drop(columns=['future_score'])
y=student_performance['future_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf=RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {rf.score(X, y)}')
pred_y = rf.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

tree=DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {tree.score(X, y)}')
pred_y = tree.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

lr=LinearRegression()
lr.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {lr.score(X, y)}')
pred_y = lr.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

gbr=GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {gbr.score(X, y)}')
pred_y = gbr.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

lasso=Lasso(random_state=42)
lasso.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {lasso.score(X, y)}')
pred_y = lasso.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

ridge=Ridge(random_state=42)
ridge.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {ridge.score(X, y)}')
pred_y = ridge.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

svr=SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {svr.score(X, y)}')
pred_y = svr.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

pca = PCA(n_components=5) 
regression = LinearRegression()
pipeline = Pipeline([('pca', pca), ('regression', regression)])

pipeline.fit(X_train, y_train)
print(f'Score : {pipeline.score(X, y)}')
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

elc=ElasticNet(random_state=42)
elc.fit(X_train, y_train)
print('\n Model training complete')

print(f'Score : {elc.score(X, y)}')
pred_y = elc.predict(X_test)

mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
accuracy = 100 - rmse
R2=r2_score(y_test, pred_y)

print(f"Accuracy: {accuracy:.2f}%")
print('R2 score :',R2)
print(f"RMSE: {rmse:.2f}")
print(f'MSE : {mse: .2f}')

import pickle

with open ('model_selected.pkl','wb') as file:
    pickle.dump(rf, file)

with open('encoder.pkl','wb') as file:
    pickle.dump(label_encode, file)

import shap 

model= pickle.load(open('model_selected.pkl', 'rb'))

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Waterfall plot for the first observation
shap.waterfall_plot(shap_values[0])

# for dropout risk detection project

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

dropout=pd.read_csv('dropout_risk.csv')

X = dropout.drop('dropout', axis=1)
y = dropout['dropout']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 2. Train XGBoost Classifier ---
print("Training XGBoost Classifier...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
print("XGBoost Classifier trained successfully.")
print("-" * 50)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] 

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print("-" * 50)

fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
print(f"False Positive Rate (FPR): {fpr:.4f}")

# False Negative Rate (FNR): FN / (FN + TP)
# It's the proportion of actual positives that were incorrectly classified as negative.
fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
print(f"False Negative Rate (FNR): {fnr:.4f}")
print("-" * 50)

# --- Analyze Model Importance (as requested in the image) ---
print("Feature Importances:")
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


import pickle

with open ('XGB_classifier.pkl','wb') as file:
    pickle.dump(model, file)




# for topic detection project

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer

topic=pd.read_csv('topic_detection.csv')

label_encoder=LabelEncoder()
int_encode=label_encoder.fit_transform(topic['label'])
y= to_categorical(int_encode)

vocab_size = 5000 # Max number of words to keep, based on word frequency
embedding_dim = 100 # Dimension of the dense embedding
maxlen = 100

tokenizer=Tokenizer(num_words=vocab_size,oov_token="<unk>")
tokenizer.fit_on_texts(topic['text'])

sequence=tokenizer.texts_to_sequences(topic['text'])

padded_sequences = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42, stratify=y)

model= Sequential([
    Embedding(vocab_size,embedding_dim,input_length=maxlen),
    Bidirectional(LSTM(128)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history=model.fit(X_train,y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

# Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1) # Get the index of the highest probability
y_true = np.argmax(y_test, axis=1) # Get the true class indices

# Decode numerical predictions back to original labels
predicted_labels = label_encoder.inverse_transform(y_pred)
true_labels = label_encoder.inverse_transform(y_true)

# Print classification report for detailed performance metrics (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))





# for digit recognizer project

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 3. Build the CNN Model
model = models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Max Pooling Layer 1
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Max Pooling Layer 2
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output of the convolutional layers to feed into dense layers
model.add(layers.Flatten())

# Dense Layer 1
model.add(layers.Dense(64, activation='relu'))
# Output Layer (10 classes for digits 0-9)
model.add(layers.Dense(10, activation='softmax'))

# Compile the Model

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# 5. Train the Model
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# 6. Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')






# for topic summarizer

from transformers import T5Tokenizer, T5ForConditionalGeneration

 
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0])

# Example usage:
text = "Binary numbers are composed of only two digits: 0 and 1. They are fundamental in computer science because digital electronic circuits are built upon this system."
summary = summarize(text)
print(summary)

# Importing evaluate library
import evaluate

# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Example sentences (non-tokenized)
reference = ["Binary numbers are composed of only two digits: 0 and 1. They are fundamental in computer science because digital electronic circuits are built upon this system."]
candidate = ["Binary numbers are composed of only two digits: 0 and 1. They are fundamental in computer science because digital electronic circuits are built upon this system."]

# BLEU expects plain text inputs
bleu_results = bleu_metric.compute(predictions=candidate, references=reference)
print(f"BLEU Score: {bleu_results['bleu'] * 100:.2f}")

# ROUGE expects plain text inputs
rouge_results = rouge_metric.compute(predictions=candidate, references=reference)

# Access ROUGE scores (no need for indexing into the result)
print(f"ROUGE-1 F1 Score: {rouge_results['rouge1']:.2f}")
print(f"ROUGE-L F1 Score: {rouge_results['rougeL']:.2f}")


# Calculating sentiment analysis scores
from textblob import TextBlob

# Example text
text_1 =  "Solving quadratic equations often involves factoring, completing the square, or using the quadratic formula. The formula, $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$, is particularly useful for complex cases."
text_2 = "Newton's laws of motion describe the relationship between a body and the forces acting upon it, and its motion in response to those forces. The first law is often called the law of inertia."

# Analyze sentiment for text_1
blob_1 = TextBlob(text_1)
polarity_1 = blob_1.sentiment.polarity
subjectivity_1 = blob_1.sentiment.subjectivity

print(f"Text 1 - Polarity: {polarity_1}, Subjectivity: {subjectivity_1}")

# Analyze sentiment for text_2
blob_2 = TextBlob(text_2)
polarity_2 = blob_2.sentiment.polarity
subjectivity_2 = blob_2.sentiment.subjectivity

print(f"Text 2 - Polarity: {polarity_2}, Subjectivity: {subjectivity_2}")
