import pandas as pd
df = pd.read_csv("/content/Dataset.csv", encoding='latin-1' )
df.head()
x=df.drop(["v1","Unnamed: 2",	"Unnamed: 3",	"Unnamed: 4"],axis=1)
y=df["v1"]
from sklearn.feature_extraction.text import TfidfVectorizer 
vector=TfidfVectorizer()
xv=vector.fit_transform(x["v2"])

from sklearn.model_selection import train_test_split 
a,b,c,d=train_test_split(xv,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression 
obj=LogisticRegression()

obj.fit(a,c) 
ypred=obj.predict(b)


from sklearn.metrics import accuracy_score 
print(accuracy_score(d,ypred))


spam_email =["Congratulations! You've won a free vacation to a luxurious resort. Click here to claim your prize now!"]
bcap=vector.transform(spam_email)
print(obj.predict(bcap))