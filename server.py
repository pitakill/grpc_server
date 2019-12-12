from concurrent import futures
import logging
import grpc

import service_pb2
import service_pb2_grpc

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class BellatorServicer(service_pb2_grpc.BellatorServicer):
	def Generis(self, request, context):
		return service_pb2.Response(Generis="test generis")

	def GenerisPolo(self, request, context):
		# Function importing Dataset
		def importdata():
			balance_data = pd.read_csv(
				'https://gist.githubusercontent.com/pitakill/2de52930db5f903d33db166133b62606/raw/f140fa2b6b590f721eec989f2c60ceb5595a9b70/data.csv',
				sep= ',',
				header = None
			)

			# Printing the dataset shape
			print ("Dataset Length: ", len(balance_data))
			print ("Dataset Shape: ", balance_data.shape)

			# Printing the dataset observations
			print ("Dataset: ",balance_data.head())
			return balance_data

		# Function to split the dataset
		def splitdataset(balance_data):
			# Separating the target variable
			X = balance_data.values[:, 1:7]
			Y = balance_data.values[:, 0]

			# Splitting the dataset into train and test
			X_train, X_test, y_train, y_test = train_test_split(
			X, Y, test_size = 0.3, random_state = 100)

			return X, Y, X_train, X_test, y_train, y_test

		# Function to perform training with entropy.
		def train_using_entropy(X_train, X_test, y_train):
			# Decision tree with entropy
			clf_entropy = DecisionTreeClassifier(
					criterion = "entropy", random_state = 100,
					max_depth = 3, min_samples_leaf = 5)

			# Performing training
			clf_entropy.fit(X_train, y_train)
			return clf_entropy


		# Function to make predictions
		def prediction(X_test, clf_object):
			y_pred = clf_object.predict(X_test)
			print("Predicted values:")
			print(y_pred)
			return y_pred

		# Function to calculate accuracy
		def cal_accuracy(y_test, y_pred):
			print("Confusion Matrix: ",
				confusion_matrix(y_test, y_pred))

			accuracy = accuracy_score(y_test,y_pred)*100
			print ("Accuracy : ", accuracy)

			print("Report : ",
			classification_report(y_test, y_pred))

			return str(accuracy) + "%"

		data = importdata()
		X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
		clf_entropy = train_using_entropy(X_train, X_test, y_train)

		# Prediction using entropy
		print("Results Using Entropy:")
		y_pred_entropy = prediction(X_test, clf_entropy)
		accuracy = cal_accuracy(y_test, y_pred_entropy)
		return service_pb2.Response(Generis=accuracy)

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	service_pb2_grpc.add_BellatorServicer_to_server(
		BellatorServicer(), server)
	server.add_insecure_port('[::]:50051')
	server.start()
	server.wait_for_termination()

if __name__ == '__main__':
	logging.basicConfig()
	serve()
