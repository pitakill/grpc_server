from concurrent import futures
import logging
import grpc

import service_pb2
import service_pb2_grpc

import joblib
import pandas as pd

from polos import importdata, splitdataset, train_using_entropy, prediction, cal_accuracy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class BellatorServicer(service_pb2_grpc.BellatorServicer):
	def Generis(self, request, context):
		# load from context the file hardcoded by now
		predictor = joblib.load("models/random_forest.pkl")
		print("here we are .....")
		# getting just one entry
		filtered = {"adds" : request.Adds,
				  "dels" : request.Dels,
				  "files": request.Files,
				  "coupling_average": request.CouplingAverage,
				  "author": request.Author,
				  }

		filtered["total_changes"] = filtered["adds"] - filtered["dels"]

		print(filtered)
		pull = pd.DataFrame([filtered, filtered])
		print(pull)
		# customizations
		cat_attribs = ["author"]
		full_pipeline = ColumnTransformer([
			("cat", OneHotEncoder(), cat_attribs)
		])

		print("toniiis -------")
		print(dir(request))
		pulls_prepared = full_pipeline.fit_transform(pull)
		print("result ....")
		print(predictor.predict(pulls_prepared))


		return service_pb2.Response(Generis="test generis")

	def GenerisPolo(self, request, context):
		# Function importing Dataset
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
