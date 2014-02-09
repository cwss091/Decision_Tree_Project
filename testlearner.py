import numpy as np
import matplotlib.pyplot as plt
import csv
import time

from KNNLearner import *
from LinRegLearner import *
from RandomForestLearner import *

def _csv_read(filename):
	'''
	import csv files which contains the data to train and test
	@filename: imported file's name
	'''

	reader = csv.reader(open(filename, 'rU'), delimiter = ',')
	Xdata = []
	Ydata = []
	for row in reader:
		Ydata.append(row[-1])
		Xdata.append(row[0:-1])
	
	Xdata = np.array(Xdata)
	Ydata = np.array(Ydata)

	return Xdata, Ydata

def RMSE(DTest, DLearn):
	'''
	Root-Mean-Square Error
	@DTest: tested correct data
	@Dlearn: data form learning
	'''
	row_queryr = float(DLearn.shape[0])
	RMSE = np.sum(np.power((DTest - DLearn), 2.0)) / row_queryr
	RMSE = np.sqrt(RMSE)
	return RMSE

def testlearner():
	'''
	test Random forest and compare with KNN
	'''

	Xdcp, Ydcp = _csv_read("data-classification-prob.csv")
	Xdrp, Ydrp = _csv_read("data-ripple-prob.csv") # the data in numpy array now is string instead of float
	
	#divide data for train and test
	dcp_row_N = Xdcp.shape[0]
	drp_row_N = Xdrp.shape[0]
	trainperct = 0.6 # data for training is 60% of total data
	dcp_trp = int(dcp_row_N * trainperct)
	drp_trp = int(drp_row_N * trainperct)
	#testperct = 1.0 - trainperct # data for test's percent 
	#data for training
	Xdcp_train = Xdcp[0:dcp_trp, :]
	Ydcp_train = np.zeros([dcp_trp, 1])
	Ydcp_train[:, 0] = Ydcp[0:dcp_trp]
	Xdrp_train = Xdrp[0:drp_trp, :]
	Ydrp_train = np.zeros([drp_trp, 1])
	Ydrp_train[:, 0] = Ydrp[0:drp_trp]
	#data for test (query)
	Xdcp_test = Xdcp[dcp_trp:dcp_row_N, :]
	Ydcp_test = np.zeros([dcp_row_N - dcp_trp, 1])
	Ydcp_test[:, 0] = Ydcp[dcp_trp:dcp_row_N]
	#Ydcp_test = [:, 0:col_n] = Xdata
	Xdrp_test = Xdrp[drp_trp:drp_row_N, :]
	Ydrp_test = np.zeros([drp_row_N - drp_trp, 1])
	Ydrp_test[:, 0] = Ydrp[drp_trp:drp_row_N]

	#print Xdcp_train

	# result of KNN learn, rows records k, training time cost, query time cost, RMSError and Correlation coeffient
	DT_dcp_result = np.zeros([5, 100]) # result of data-classification-prob.csv of RF
	DT_drp_result = np.zeros([5, 100]) # result of data-ripple-prob.csv of RF
	KNN_dcp_result = np.zeros([2, 100]) # results of data-classification-prob.csv of KNN
	KNN_drp_result = np.zeros([2, 100]) # results of data-ripple-prob.csv of KNN

	#print len(RFL.trees)
	for k in range(1, 101):
		#k = 30
		# Random forest learner
		RFL = RandomForestLearner(k)
		KNN_lner = KNNLearner(k)
		
		DT_dcp_result[0][k-1] = k
		DT_drp_result[0][k-1] = k
		# result of data-classification-prob
		stime = time.time()
		RFL.addEvidence(Xdcp_train, Ydcp_train)
		etime = time.time()
		DT_dcp_result[1][k-1] = etime - stime

		KNN_lner.addEvidence(Xdcp_train, Ydcp_train)

		#print len(RFL.trees)
		#RFL.trees[0].print_tree(RFL.trees[0].root)
		stime = time.time()
		Ydcp_learn = RFL.query(Xdcp_test)
		etime = time.time()
		DT_dcp_result[2][k-1] = etime - stime;

		Ydcp_learn_KNN = KNN_lner.query(Xdcp_test)

		DT_dcp_result[3][k-1] = RMSE(Ydcp_learn, Ydcp_test)
		KNN_dcp_result[0][k-1] = RMSE(Ydcp_learn_KNN, Ydcp_test)

		DT_dcp_result[4][k-1] = np.corrcoef(Ydcp_learn.T, Ydcp_test.T)[0][1]
		KNN_dcp_result[1][k-1] = np.corrcoef(Ydcp_learn_KNN.T, Ydcp_test.T)[0][1]

		# result of data-ripple
		#RFL1 = RandomForestLearner(k)
		stime = time.time()
		RFL.addEvidence(Xdrp_train, Ydrp_train)
		etime = time.time()
		DT_drp_result[1][k-1] = etime - stime

		KNN_lner.addEvidence(Xdrp_train, Ydrp_train)

		#print len(RFL.trees)
		#RFL.trees[0].print_tree(RFL.trees[0].root)
		stime = time.time()
		Ydrp_learn = RFL.query(Xdrp_test)
		etime = time.time()
		DT_drp_result[2][k-1] = etime - stime;

		Ydrp_learn_KNN = KNN_lner.query(Xdrp_test)

		#print Ydrp_learn_KNN

		DT_drp_result[3][k-1] = RMSE(Ydrp_learn, Ydrp_test)
		KNN_drp_result[0][k-1] = RMSE(Ydrp_learn_KNN, Ydrp_test)

		DT_drp_result[4][k-1] = np.corrcoef(Ydrp_learn.T, Ydrp_test.T)[0][1]
		KNN_drp_result[1][k-1] = np.corrcoef(Ydrp_learn_KNN.T, Ydrp_test.T)[0][1]
		#print DT_drp_result[4][k-1]
	
	plt.clf()
	fig = plt.figure()
	fig.suptitle('RMS Error of Classification data test')
	plt.plot(DT_dcp_result[0, :], DT_dcp_result[3, :], 'r', label = 'Random Forest')
	plt.plot(DT_dcp_result[0, :], KNN_dcp_result[0, :], 'b', label = 'KNN')
	plt.legend(loc = 1)
	plt.xlabel('K')
	plt.ylabel('RMS Error')
	fig.savefig('classification-RMSE.pdf', format = 'pdf')

	plt.clf()
	fig = plt.figure()
	fig.suptitle('Correlation Coefficient of Classification data test')
	plt.plot(DT_dcp_result[0, :], DT_dcp_result[4, :], 'r', label = 'Random Forest')
	plt.plot(DT_dcp_result[0, :], KNN_dcp_result[1, :], 'b', label = 'KNN')
	plt.legend(loc = 4)
	plt.xlabel('K')
	plt.ylabel('Correlation Coefficient')
	fig.savefig('classification-Corr.pdf', format = 'pdf')

	plt.clf()
	fig = plt.figure()
	fig.suptitle('RMS Error of Ripple data test')
	plt.plot(DT_drp_result[0, :], DT_drp_result[3, :], 'r', label = 'Random Forest')
	plt.plot(DT_drp_result[0, :], KNN_drp_result[0, :], 'b', label = 'KNN')
	plt.legend(loc = 2)
	plt.xlabel('K')
	plt.ylabel('RMS Error')
	fig.savefig('ripple-RMSE.pdf', format = 'pdf')

	plt.clf()
	fig = plt.figure()
	fig.suptitle('Correlation Coefficient of Ripple data test')
	plt.plot(DT_drp_result[0, :], DT_drp_result[4, :], 'r', label = 'Random Forest')
	plt.plot(DT_drp_result[0, :], KNN_drp_result[1, :], 'b', label = 'KNN')
	plt.legend(loc = 3)
	plt.xlabel('K')
	plt.ylabel('Correlation Coefficient')
	fig.savefig('ripple-Corr.pdf', format = 'pdf')




	# plot the Y data of ripple data
	#plt.clf()
	#fig = plt.figure()
	#fig.suptitle('Y of classification data')
	#plt.scatter(Ydcp_test, Ydcp_learn)
	#plt.xlabel('Actual Y')
	#plt.ylabel('Predicted Y')
	#fig.savefig('classification_Y.pdf', format = 'pdf')

if __name__ == "__main__":
	testlearner()


	
