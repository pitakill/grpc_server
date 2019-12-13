from concurrent import futures
import logging
import grpc

import service_pb2
import service_pb2_grpc

import joblib
import pandas as pd
import numpy as np

from polos import importdata, splitdataset, train_using_entropy, prediction, cal_accuracy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

adds_ix, dels_ix = 3, 4

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        net_changes = X[:, adds_ix] - X[:, dels_ix]
        return np.c_[X, net_changes]

class BellatorServicer(service_pb2_grpc.BellatorServicer):
	def Generis(self, request, context):
		# load from context the file hardcoded by now
		predictor = joblib.load("models/random_forest.pkl")
		filtered = {
            "comments": request.Comments,
            "open_days": request.OpenDays,
			"author": request.Author,
            "adds" : request.Adds,
			"dels" : request.Dels,
            "files": request.Files,
            "total_changes": 0,
			"coupling_average": request.CouplingAverage,
            "tag": "normal",
            "changes_size": 0
		}
		# some hack just because is almost 1 am
		filtered["total_changes"] = filtered["adds"] - filtered["dels"]
		pull = pd.DataFrame([filtered, filtered])
		pull["total_changes"]
		pull["changes_size"] = pd.cut(pull["total_changes"], bins=[250, 500, 1000, 5000,  np.inf], labels=[1, 2, 3, 4])
		pulls_prepared = predictor["pipeline"].transform(pull)
		print("result ....")
		results = predictor["model"].predict(pulls_prepared)

		return service_pb2.Response(Generis="%.2f." % results[0])

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
