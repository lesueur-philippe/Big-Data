import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.neural_network import MLPClassifier

class Train_master_learning:

	tfidf_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	mlp_clf = MLPClassifier(activation = 'logistic', random_state=42)
	score = 0

	def __init__(self):
		df = self.load_data("../train_master.csv")
		self.add_length(df)
		self.add_NBupper(df)
		self.add_NBlower(df)
		self.add_NBponct(df)
		self.add_NBnumber(df)
		self.add_NBsentences(df)
		df = df.dropna()
		X_train, X_validation, X_test, y_train, y_validation , y_test = self.train_validation_test_split(df.drop(["Target","ID", "product"],axis = 1),df.Target)
		self.tfidf_clf.fit(X_train.review_content, y_train)
		X_train["text_clf"] = self.tfidf_clf.predict(X_train.review_content)
		X_validation["text_clf"] = self.tfidf_clf.predict(X_validation.review_content)
		X_test["text_clf"] = self.tfidf_clf.predict(X_test.review_content)
		self.mlp_clf.fit(X_train.drop(["review_content","review_title"],axis=1), y_train)
		score_val = self.mlp_clf.score(X_validation.drop(["review_content","review_title"],axis=1), y_validation)
		score_test = self.mlp_clf.score(X_test.drop(["review_content","review_title"],axis=1), y_test)
		self.score = (score_val + score_test)/2
		
		
	def load_data(self, data_path):
		df = pd.read_csv(data_path)
		if "Unnamed: 0" in df.columns:
			df = df.drop("Unnamed: 0",axis=1)
		return df
	
	def add_FirstClassif(self, df):
		df["text_clf"] = self.tfidf_clf.predict(df.review_content)
		return df
	
	def add_length(self, df):
		df["review_length"] = df.review_content.str.len()
		df["title_length"] = df.review_title.str.len()
		return df

	def add_NBupper(self, df):
		df["review_NBupper"] = df.review_content.str.findall(r'[A-Z]').str.len()
		df["title_NBupper"] = df.review_title.str.findall(r'[A-Z]').str.len()
		return df

	def add_NBlower(self, df):
		df["review_NBlower"] = df.review_content.str.findall(r'[a-z]').str.len()
		df["title_NBlower"] = df.review_title.str.findall(r'[a-z]').str.len()
		return df

	def add_NBponct(self, df):
		df["review_NBponct"] = df.review_content.str.findall(r'[.!,;:\?\-\"\'\(\)]').str.len()
		df["title_NBponct"] = df.review_title.str.findall(r'[.!,;:\?\-\"\'\(\)]').str.len()
		return df
		
	def add_NBnumber(self, df):
		df["review_NBnumber"] = df.review_content.str.findall(r'[0-9]').str.len()
		df["title_NBnumber"] = df.review_title.str.findall(r'[0-9]').str.len()
		return df
		
	def add_NBsentences(self, df):
		df["review_NBsentences"] = df.review_content.str.split(r'[.!\?]').str.len()
		return df
		
	def train_validation_test_split(self, X,y, train_size=0.6, test_size=0.2):
		data_train, data_tv, y_train, y_tv = train_test_split(X, y, test_size=1-train_size, random_state=42)
		data_validation, data_test, y_validation , y_test = train_test_split(data_tv, y_tv, test_size= test_size/(1-train_size), random_state=42)
		return data_train, data_validation, data_test, y_train, y_validation , y_test

	def predict(self, X):
		self.add_length(X)
		self.add_NBupper(X)
		self.add_NBlower(X)
		self.add_NBponct(X)
		self.add_NBnumber(X)
		self.add_NBsentences(X)
		self.add_FirstClassif(X)
		return self.mlp_clf.predict(X.drop(["review_content", "review_title"],axis=1))