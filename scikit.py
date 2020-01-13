from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items

classifier = OneVsRestClassifier(SVC(kernel='linear')) #poly  94%
vectorizer = TfidfVectorizer()

# train
vectorize_text = vectorizer.fit_transform(train_data.v2)
classifier.fit(vectorize_text, train_data.v1)

vectorize_text = vectorizer.transform(test_data.v2)


for d,a in zip(vectorize_text[:30], test_data.v1[:30]):
    print('prediction:' + str(classifier.predict(d)[0]) + ' real:' + str(a))

score = classifier.score(vectorize_text, test_data.v1)
print(score) # 98,8

#print(vectorizer.get_feature_names())
#print(vectorize_text[1])

input('Press ENTER to exit') 