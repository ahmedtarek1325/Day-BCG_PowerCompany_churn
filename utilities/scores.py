from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt


def report_scores(target,predict):
	accracy= accuracy_score(target,predict)*100
	precision = precision_score(target,predict,average='weighted')*100
	recall= recall_score(target,predict,average='weighted')*100
	auc= roc_auc(target,predict)*100
	f1_weigthed= f1_score(target,predict,average='weighted')*100


	print(f'Accuracy Score: {accracy:0.2f}',end=" || ")
	print(f'precision_weighted accuracy: {precision:0.2f} ',end=" || ")
	print(f'recall_weighted accuracy {recall:0.2f}',end=" || ")
	print(f"f1_weighted {f1_weigthed:0.2f}",end=" || ")
	print(f"Area under curve {auc:0.2f}")
	
	#print("\n")
	print('Confusion matrix ')
	print(confusion_matrix(target,predict))
	
	return accracy,precision,recall,f1_weigthed,auc


def roc_auc(target,predicted):
	return roc_auc_score(target, predicted, average='weighted')



def plot_roc(target,predicted): 
	'''
	INPUT
	- take target varaibles 
	- predicaed variables
	OUTPUTs 
	- NONE
	ACTIONS 
	PLot ROC curve
	'''
	
	fpr,tpr,threshold= roc_curve(target,predicted)
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc(target,predicted))
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()
	return 