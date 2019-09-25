#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:35:52 2019

@author: le86qiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import os

deepipred_results_dir = f'/home/go96bix/projects/raw_data/non_binary_250_nodes_1000epochs/results'

# read test file table
#testfiletable = '/home/go96bix/projects/epitop_pred/data_generator_bepipred_non_binary/samples_for_ROC.csv'
testfiletable = '/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/samples_for_ROC.csv'
testproteinIDs = []
with open(testfiletable) as infile:
	for line in infile:
		file = line.strip().rsplit('/',1)[1]
		testproteinIDs.append(file[:-6])

# get start/stop postions of epitopes/nonepitopes
startstop_epi = {}
startstop_nonepi = {}
for testid in testproteinIDs:
	file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred_proteins_with_marking/{testid}.fasta'
	with open(file) as infile:
		for line in infile:
			line = line[1:].strip().split()
			for epiID in line:
				epiID = epiID.split('_')
				flag = epiID[0]
				start = int(epiID[2])
				stop = int(epiID[3])
				if flag == 'PositiveID':
					if testid in startstop_epi:
						startstop_epi[testid].append([start,stop])
					else:
						startstop_epi[testid] = [[start,stop]]
					#startstop_epi.update({proteinID: startstop_epi.get(proteinID,[]).append([start,stop])})
				else:
					if testid in startstop_nonepi:
						startstop_nonepi[testid].append([start,stop])
					else:
						startstop_nonepi[testid] = [[start,stop]]
					#startstop_nonepi.update({proteinID: startstop_nonepi.get(proteinID,[]).append([start,stop])})
			break


# read bepipred
bepipred_scores = []
bepipred_flag = []
for testid in testproteinIDs:
	bepipred_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred/results/{testid}.csv'
	bepipred_table = pd.read_csv(bepipred_file, sep="\t", index_col=None, skiprows = 1).values
	bepipred_table = bepipred_table[:,7]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = bepipred_table[start:stop]
		score = sum(scores) / len(scores)
		bepipred_scores.append(score)
		bepipred_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = bepipred_table[start:stop]
		score = sum(scores) / len(scores)
		bepipred_scores.append(score)
		bepipred_flag.append(0)
bepipred_scores = np.array(bepipred_scores)
bepipred_flag = np.array(bepipred_flag)


#testproteinIDs = ['protein_1032']

# read antigenicity
antigenicity_scores = []
antigenicity_flag = []
for testid in testproteinIDs:
	antigenicity_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/antigenicity/{testid}.csv'
	antigenicity_table = pd.read_csv(antigenicity_file, sep="\t", index_col=None).values
	antigenicity_table = antigenicity_table[:,1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = antigenicity_table[start:stop]
		score = sum(scores) / len(scores)
		antigenicity_scores.append(score)
		antigenicity_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = antigenicity_table[start:stop]
		score = sum(scores) / len(scores)
		antigenicity_scores.append(score)
		antigenicity_flag.append(0)
antigenicity_scores = np.array(antigenicity_scores)
antigenicity_flag = np.array(antigenicity_flag)

# read betaturn
betaturn_scores = []
betaturn_flag = []
for testid in testproteinIDs:
	betaturn_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/betaturn/{testid}.csv'
	betaturn_table = pd.read_csv(betaturn_file, sep="\t", index_col=None).values
	betaturn_table = betaturn_table[:,1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = betaturn_table[start:stop]
		score = sum(scores) / len(scores)
		betaturn_scores.append(score)
		betaturn_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = betaturn_table[start:stop]
		score = sum(scores) / len(scores)
		betaturn_scores.append(score)
		betaturn_flag.append(0)
betaturn_scores = np.array(betaturn_scores)
betaturn_flag = np.array(betaturn_flag)

# read hydrophilicity
hydrophilicity_scores = []
hydrophilicity_flag = []
for testid in testproteinIDs:
	hydrophilicity_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/hydrophilicity/{testid}.csv'
	hydrophilicity_table = pd.read_csv(hydrophilicity_file, sep="\t", index_col=None).values
	hydrophilicity_table = hydrophilicity_table[:,1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = hydrophilicity_table[start:stop]
		score = sum(scores) / len(scores)
		hydrophilicity_scores.append(score)
		hydrophilicity_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = hydrophilicity_table[start:stop]
		score = sum(scores) / len(scores)
		hydrophilicity_scores.append(score)
		hydrophilicity_flag.append(0)
hydrophilicity_scores = np.array(hydrophilicity_scores)
hydrophilicity_flag = np.array(hydrophilicity_flag)

# read accessibility
accessibility_scores = []
accessibility_flag = []
for testid in testproteinIDs:
	accessibility_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/accessibility/{testid}.csv'
	accessibility_table = pd.read_csv(accessibility_file, sep="\t", index_col=None).values
	accessibility_table = accessibility_table[:,1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = accessibility_table[start:stop]
		score = sum(scores) / len(scores)
		accessibility_scores.append(score)
		accessibility_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = accessibility_table[start:stop]
		score = sum(scores) / len(scores)
		accessibility_scores.append(score)
		accessibility_flag.append(0)
accessibility_scores = np.array(accessibility_scores)
accessibility_flag = np.array(accessibility_flag)

# read deepipred
deepipred_scores = []
deepipred_flag = []
for testid in testproteinIDs:
	deepipred_file = f'{os.path.join(deepipred_results_dir,"deepipred/")}{testid}.csv'
	# deepipred_file = f'/home/go96bix/projects/raw_data/non_binary_100_nodes_400epochs_only_test/results/deepipred/{testid}.csv'
	# deepipred_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/deepipred/results/deepipred/{testid}.csv'
	deepipred_table = pd.read_csv(deepipred_file, sep="\t", index_col=None).values
	deepipred_table = deepipred_table[:,1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = deepipred_table[start:stop]
		score = sum(scores) / len(scores)
#		score = scores[len(scores) % 2 + 5]
		deepipred_scores.append(score)
		deepipred_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = deepipred_table[start:stop]
		score = sum(scores) / len(scores)
		deepipred_scores.append(score)
		deepipred_flag.append(0)
deepipred_scores = np.array(deepipred_scores)
deepipred_flag = np.array(deepipred_flag)

# read raptorx
raptorx_scores = []
raptorx_flag = []
for testid in testproteinIDs:
	raptorx_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/raptorx/results/flo_files/{testid}.csv'
	try:
		raptorx_table = pd.read_csv(raptorx_file, sep="\t", index_col=None).values
	except:
		continue
	iupred_table = raptorx_table[:,6]
	structure_table = raptorx_table[:,2] - raptorx_table[:,1] - raptorx_table[:,0]		# coil - helix - sheet
	accessibility_table = raptorx_table[:,5] - raptorx_table[:,3]		# exposed - bury
	raptorx_table = structure_table + accessibility_table
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = raptorx_table[start:stop]
		score = sum(scores) / len(scores)
		raptorx_scores.append(score)
		raptorx_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = raptorx_table[start:stop]
		score = sum(scores) / len(scores)
		raptorx_scores.append(score)
		raptorx_flag.append(0)
raptorx_scores = np.array(raptorx_scores)
raptorx_flag = np.array(raptorx_flag)

# read iupred
iupred_scores = []
iupred_flag = []
for testid in testproteinIDs:
	iupred_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/raptorx/results/iupred_results/{testid}_iupred.res'
	iupred_table = pd.read_csv(iupred_file, sep="\s+", index_col=None, skiprows = 9,header = None).values
	iupred_table = iupred_table[:,2]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = iupred_table[start:stop]
		score = sum(scores) / len(scores)
		iupred_scores.append(score)
		iupred_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = iupred_table[start:stop]
		score = sum(scores) / len(scores)
		iupred_scores.append(score)
		iupred_flag.append(0)
iupred_scores = np.array(iupred_scores)
iupred_flag = np.array(iupred_flag)



# calculate roc curve
from sklearn.metrics import roc_curve, auc

thresh = 1
fpr = {}
tpr = {}
roc_auc = {}
thresholds = {}

key = 'bepipred'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(bepipred_flag, bepipred_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(bepipred_flag,bepipred_scores,max_fpr=thresh)
key = 'antigenicity'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(antigenicity_flag, antigenicity_scores, pos_label=0)
roc_auc[key] = 1 -metrics.roc_auc_score(antigenicity_flag,antigenicity_scores,max_fpr=thresh)
key = 'hydrophilicity'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(hydrophilicity_flag, hydrophilicity_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(hydrophilicity_flag,hydrophilicity_scores,max_fpr=thresh)
key = 'accessibility'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(accessibility_flag, accessibility_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(accessibility_flag,accessibility_scores,max_fpr=thresh)
key = 'betaturn'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(betaturn_flag, betaturn_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(betaturn_flag,betaturn_scores,max_fpr=thresh)
key = 'deepipred'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(deepipred_flag, deepipred_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(deepipred_flag,deepipred_scores,max_fpr=thresh)
key = 'raptorx'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(raptorx_flag, raptorx_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(raptorx_flag,raptorx_scores,max_fpr=thresh)
key = 'iupred'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(iupred_flag, iupred_scores, pos_label=1)
roc_auc[key] = metrics.roc_auc_score(iupred_flag,iupred_scores,max_fpr=thresh)



# plot
plt.figure(figsize = (6,6))
lw = 2
for i in fpr:
	fpr[i] = [x for x in fpr[i] if x <= thresh]
	tpr[i] = tpr[i][:len(fpr[i])]
maxtpr = 0
for x in tpr:
	maxtpr = max(maxtpr,max(tpr[x]))
plt.plot(fpr['deepipred'], tpr['deepipred'], color='green', lw=lw, label='DeEpiPred (area = %0.2f)' % roc_auc['deepipred'])
plt.plot(fpr['bepipred'], tpr['bepipred'], color='grey', lw=lw, label='Bepipred (area = %0.2f)' % roc_auc['bepipred'])
plt.plot(fpr['antigenicity'], tpr['antigenicity'], color='darkgrey', linestyle = ':', lw=lw, label='Antigenicity-avg (area = %0.2f)' % roc_auc['antigenicity'])
plt.plot(fpr['hydrophilicity'], tpr['hydrophilicity'], color='darkgrey', lw=lw, label='Hydrophilicity-avg (area = %0.2f)' % roc_auc['hydrophilicity'])
plt.plot(fpr['accessibility'], tpr['accessibility'], color='darkgrey',linestyle = '--', lw=lw, label='Accessibility-avg (area = %0.2f)' % roc_auc['accessibility'])
plt.plot(fpr['betaturn'], tpr['betaturn'], color='darkgrey', linestyle = '-.', lw=lw, label='Betaturn-avg (area = %0.2f)' % roc_auc['betaturn'])
plt.plot(fpr['raptorx'], tpr['raptorx'], color='orange', linestyle = '--', lw=lw, label='RaptorX (area = %0.2f)' % roc_auc['raptorx'])
plt.plot(fpr['iupred'], tpr['iupred'], color='grey', linestyle = ':', lw=lw, label='IUPred (area = %0.2f)' % roc_auc['iupred'])

plt.plot([0, thresh], [0, thresh], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, thresh])
plt.ylim([0.0, 1.0 * maxtpr])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.show()
#plt.savefig(os.path.join(deepipred_results_dir,f"ROC_prediction_comparison_{thresh}.pdf"),bbox_inches="tight", pad_inches=0)
plt.savefig(f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/ROC_prediction_comparison_{thresh}.pdf',bbox_inches="tight", pad_inches=0)
plt.close()

# calculate precision-recall curve

precision = {}
recall = {}
thresholds = {}

key = 'bepipred'
precision[key], recall[key], thresholds[key] = precision_recall_curve(bepipred_flag, bepipred_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'antigenicity'
precision[key], recall[key], thresholds[key] = precision_recall_curve(antigenicity_flag, antigenicity_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'hydrophilicity'
precision[key], recall[key], thresholds[key] = precision_recall_curve(hydrophilicity_flag, hydrophilicity_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'accessibility'
precision[key], recall[key], thresholds[key] = precision_recall_curve(accessibility_flag, accessibility_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'betaturn'
precision[key], recall[key], thresholds[key] = precision_recall_curve(betaturn_flag, betaturn_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'deepipred'
precision[key], recall[key], thresholds[key] = precision_recall_curve(deepipred_flag, deepipred_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'raptorx'
precision[key], recall[key], thresholds[key] = precision_recall_curve(raptorx_flag, raptorx_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])
key = 'iupred'
precision[key], recall[key], thresholds[key] = precision_recall_curve(iupred_flag, iupred_scores, pos_label=1)
roc_auc[key] = auc(recall[key],precision[key])

# plot
plt.figure(figsize = (6,6))
lw = 2
thresh = 1
for i in recall:
	recall[i] = [x for x in recall[i] if x <= thresh]
	precision[i] = precision[i][:len(recall[i])]
maxtpr = 0
for x in precision:
	maxtpr = max(maxtpr,max(precision[x]))
plt.plot(recall['deepipred'], precision['deepipred'], color='green', lw=lw, label='DeEpiPred (area = %0.2f)' % roc_auc['deepipred'])
plt.plot(recall['bepipred'], precision['bepipred'], color='grey', lw=lw, label='Bepipred (area = %0.2f)' % roc_auc['bepipred'])
plt.plot(recall['antigenicity'], precision['antigenicity'], color='darkgrey', linestyle = ':', lw=lw, label='Antigenicity-avg (area = %0.2f)' % roc_auc['antigenicity'])
plt.plot(recall['hydrophilicity'], precision['hydrophilicity'], color='darkgrey', lw=lw, label='Hydrophilicity-avg (area = %0.2f)' % roc_auc['hydrophilicity'])
plt.plot(recall['accessibility'], precision['accessibility'], color='darkgrey',linestyle = '--', lw=lw, label='Accessibility-avg (area = %0.2f)' % roc_auc['accessibility'])
plt.plot(recall['betaturn'], precision['betaturn'], color='darkgrey', linestyle = '-.', lw=lw, label='Betaturn-avg (area = %0.2f)' % roc_auc['betaturn'])
plt.plot(recall['raptorx'], precision['raptorx'], color='orange', linestyle = '--', lw=lw, label='RaptorX (area = %0.2f)' % roc_auc['raptorx'])
plt.plot(recall['iupred'], precision['iupred'], color='grey', linestyle = ':', lw=lw, label='IUPred (area = %0.2f)' % roc_auc['iupred'])

ratio_true_false = deepipred_flag.sum()/len(deepipred_flag)
plt.plot([0, 1], [ratio_true_false, ratio_true_false], color='navy', linestyle='--', label='random (area = %0.2f)' % ratio_true_false)
plt.xlim([0.0, thresh])
plt.ylim([0.0, 1.0 * maxtpr])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
#plt.show()
#plt.savefig(os.path.join(deepipred_results_dir,f"precision_recall_comparison_{thresh}.pdf"),bbox_inches="tight", pad_inches=0)
plt.savefig(f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/precision_recall_comparison_{thresh}.pdf',bbox_inches="tight", pad_inches=0)
plt.close()

yPred_0 = deepipred_scores[deepipred_flag == 0]
yPred_1 = deepipred_scores[deepipred_flag == 1]
yPred_total = [yPred_0, yPred_1]

plt.hist(yPred_total, bins=20, range=(0, 1), stacked=False, label=['no Epitope', 'true Epitope'])
plt.legend()
#plt.savefig( os.path.join(deepipred_results_dir,f"prediction_distribution.pdf"))
plt.savefig("/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/prediction_distribution.pdf")
plt.close()

