import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sklearn.ensemble as sk

documents = pd.read_csv("shuffled-full-set-hashed.csv",names =['label','words'])

documents['class_labels'] = documents['label'].rank(method='dense',ascending=True).astype(int)
labelMapping=pd.Series(documents.label.values,index=documents.class_labels).to_dict()

OCRData,labels = documents.words,documents.class_labels

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=None)

featureSet = tfidfconverter.fit_transform(documents['words'].values.astype('U'))

with open("TDF_final", 'wb') as handle:
    pickle.dump(tfidfconverter, handle)

features = featureSet.toarray()
X_train, X_test, y_train, y_test = train_test_split(features,labels , test_size=0.2, random_state=0)

classifier = sk.RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)


with open('mappingDictionary', 'wb') as picklefile:
    pickle.dump(labelMapping,picklefile)


with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
