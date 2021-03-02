from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.animation import FuncAnimation
import matplotlib 
from matplotlib import rc
matplotlib.rc('xtick', labelsize=24) 
matplotlib.rc('ytick', labelsize=24) 
plt.rcParams.update({'font.size': 22})
rc('text', usetex=True)

# Functions
def saveGit(name, variableAnimate, color, fig):
	fig.set_tight_layout(True)
	ax = plt.axes()
	lineList = []
	print("len(variableAnimate): ", len(variableAnimate))
	for i in range(0, len(variableAnimate)):
		print(i)
		line, = ax.plot([], [], color[i], zorder=1)
		lineList.append(line)
		print("======== Here =========== ")
	# plt.legend(fontsize=24)
	plt.plot([0,50],[0,0], '-k')

	dataPoints = variableAnimate[0].shape[0]

	def update(i):
		print(i, "/", dataPoints)
		for j in range(0, len(variableAnimate)):
			lineList[j].set_data([variableAnimate[j][i,0], variableAnimate[j][i,1]], [variableAnimate[j][i,2], variableAnimate[j][i,3]])
			ax.set(xlim=(-0.5, 20.5), ylim=(0,1.2))


	anim = FuncAnimation(fig, update, frames=np.arange(0, dataPoints), interval=50)
	anim.save(name+'.gif', dpi=80, writer='imagemagick')

	plt.show()

def plotCL(folder, it, color, legend, marker, pltLabel=1, pltCL=1):
	if pltCL == 1:
		xtArray = load(folder+'xtArray_it'+str(it)+'.npy')
		stArray = load(folder+'stArray_it'+str(it)+'.npy')
		region  = load(folder+'region_it'+str(it)+'.npy')
	else:
		xtArray = load(folder+'ftocp_xPred.npy')
		stArray = load(folder+'ftocp_sPred.npy')
		region  = load(folder+'region.npy')

	if pltLabel == 1:
		plt.plot(xtArray[:,0], xtArray[:,1], '-', color=color, linewidth=3, label=legend)
	else:
		plt.plot(xtArray[:,0], xtArray[:,1], '-', color=color, linewidth=3)

	footLegend = legend +' step'
	for i in range(0, xtArray.shape[0]-1):
		if region[i] == 0:
			if (i == 0) and pltLabel == 1:
				plt.plot(stArray[i,0], 0, marker, color=color, label=footLegend)
			else:
				plt.plot(stArray[i,0], 0, marker, color=color)
			# print("SS left leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
		elif region[i] == 1:
			if (i == 0) and pltLabel == 1:
				plt.plot(stArray[i,0], 0, marker, color=color, label=footLegend)
			else:
				plt.plot(stArray[i,1], 0, marker, color=color)
			# print("SS right leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
		else:
			if (i == 0) and pltLabel == 1:
				plt.plot(stArray[i,0], 0, marker, color=color, label=footLegend)
			else:
				plt.plot(stArray[i,0], 0, marker, color=color)
				plt.plot(stArray[i,1], 0, marker, color=color)

def pltCL_Color(folder,it, cl=0):
	if cl == 0:
		ftocp_sPred = load(folder+'ftocp_sPred.npy')
		ftocp_xPred = load(folder+'ftocp_xPred.npy')
		ftocp_region = load(folder+'region.npy')
	else:
		ftocp_sPred = load(folder+'stArray_it'+str(it)+'.npy')
		ftocp_xPred = load(folder+'xtArray_it'+str(it)+'.npy')
		ftocp_region = load(folder+'region_it'+str(it)+'.npy')
	
	toPlot = []
	color = ['r', 'b', 'k']
	c = 0
	for i in range(0, ftocp_xPred.shape[0]):
		if ftocp_region[i] == 0:
			toPlot.append(ftocp_xPred[i,:])
			# plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'or')
			plt.plot(ftocp_sPred[i,0], 0, 'ok')
			# print("SS left leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
		elif ftocp_region[i] == 1:
			toPlot.append(ftocp_xPred[i,:])
			# plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'ob')
			plt.plot(ftocp_sPred[i,1], 0, 'ok')
			# print("SS right leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
		else:
			toPlot.append(ftocp_xPred[i,:])
			# plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'oy')
			plt.plot(ftocp_sPred[i,0], 0, 'ok')
			plt.plot(ftocp_sPred[i,1], 0, 'ok')
			# print("DS, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
		if (i>1 and ftocp_region[i] != ftocp_region[i-1]) or (i == ftocp_xPred.shape[0]-1):
			arrayToPlot = np.array(toPlot)
			if c < 3:
				c += 1
				if ftocp_region[i-1]==0:
					plt.plot(arrayToPlot[:,0], arrayToPlot[:,1], color=color[ftocp_region[i-1]], linewidth=3, label='$i = l$')
				elif ftocp_region[i-1] == 1:
					plt.plot(arrayToPlot[:,0], arrayToPlot[:,1], color=color[ftocp_region[i-1]], linewidth=3, label='$i = r$')
				else:
					plt.plot(arrayToPlot[:,0], arrayToPlot[:,1], color=color[ftocp_region[i-1]], linewidth=3, label='$i = ds$')
			else:
				plt.plot(arrayToPlot[:,0], arrayToPlot[:,1], color=color[ftocp_region[i-1]], linewidth=3)
			toPlot=[]
			toPlot.append(ftocp_xPred[i,:])

	plt.plot(ftocp_sPred[0,0], 0, 'ok', label='Step')
	plt.legend(loc="upper left", ncol=4)

	# for i in [0, 77, 170, ftocp_xPred.shape[0]-1]:
	# 	if ftocp_region[i] == 0:
	# 		# plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'or')
	# 		plt.plot(ftocp_sPred[i,0], 0, 'ok')
	# 		plt.plot([ftocp_sPred[i,0], ftocp_xPred[i,0]], [0, ftocp_xPred[i,1]], '--k')
	# 		plt.plot([ftocp_sPred[i,1], ftocp_xPred[i,0]], [0.05, ftocp_xPred[i,1]], '--k')
			
	# 		# print("SS left leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
	# 	elif ftocp_region[i] == 1:
	# 		# plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'ob')
	# 		plt.plot(ftocp_sPred[i,1], 0, 'ok')
	# 		plt.plot([ftocp_sPred[i,0], ftocp_xPred[i,0]], [0.05, ftocp_xPred[i,1]], '--k')
	# 		plt.plot([ftocp_sPred[i,1], ftocp_xPred[i,0]], [0, ftocp_xPred[i,1]], '--k')
	# 		# print("SS right leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
	# 	else:
	# 		# plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'oy')
	# 		plt.plot([ftocp_sPred[i,0], ftocp_xPred[i,0]], [0, ftocp_xPred[i,1]], '--k')
	# 		plt.plot([ftocp_sPred[i,1], ftocp_xPred[i,0]], [0, ftocp_xPred[i,1]], '--k')
	# 		plt.plot(ftocp_sPred[i,0], 0, 'ok')
	# 		plt.plot(ftocp_sPred[i,1], 0, 'ok')
	# 		# print("DS, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])

	# plt.plot(ftocp_xPred[:,0], ftocp_xPred[:,1], '-k')
	plt.xlabel("$p_x$ [m]")
	plt.ylabel("$p_y$ [m]")

# Main loop
def main():
	# Result section performance
	fig = plt.figure()
	fig.set_size_inches(14, 8)

	for i in range(0, 11):
		folder = 'data_dist_0_N_30_idxIC_'+str(i)+'/'
		plotCL(folder, 0, 'b', 'Closed-loop', '*', pltLabel=i, pltCL=1)

	plotCL(folder, 0, 'k', 'Offline trajectory', 's', pltLabel=1, pltCL=0)
	
	plt.xlabel("$p_x$ [m]")
	plt.ylabel("$p_y$ [m]")
	plt.ylim(-0.01, 0.8)
	plt.legend(loc="upper left", ncol=2)
	plt.show()


	fig = plt.figure()
	fig.set_size_inches(14, 8)
	folder = 'data_dist_0_N_30_idxIC_0/'
	it = 0
	pltCL_Color(folder,it)
	plt.ylim(-0.01, 0.7)
	

	# Comp Time
	fig = plt.figure()
	# fig.set_size_inches(14, 8)
	for i in range(0,11):
		folder = 'data_dist_0_N_30_idxIC_'+str(i)+'/'
		xtArray = load(folder+'xtArray_it0.npy')
		compTime = load(folder+'compTime_it0.npy')
		plt.plot(compTime, '-', label=str(i))
	plt.legend()
	plt.ylabel("Solver Time [s]")
	plt.xlabel("Time step")
	plt.show()

	fig = plt.figure()
	fig.set_size_inches(14, 8)
	# Dist
	folder = 'tracking_data_dist_1_N_30_idxIC_0/'
	plotCL(folder, 0, 'k', 'Offline', 's', pltLabel=1, pltCL=0)
	plotCL(folder, 0, 'r', 'Tracking', '*', pltLabel=1, pltCL=1)
	folder = 'data_dist_1_N_30_idxIC_0/'
	plotCL(folder, 0, 'b', 'Proposed', 'o', pltLabel=1, pltCL=1)
	plt.ylim(-0.01, 0.85)
	plt.xlabel("$p_x$ [m]")
	plt.ylabel("$p_y$ [m]")
	plt.legend(loc="upper left", ncol=3)
	plt.show()


	it    = 40
	itmax = 40
	N  = 30
	
	plt.figure()
	folder = 'data_N_'+str(N)+'/'

	cost =[]
	timeSteps = []
	itTot = []
	for i in range(0, itmax+1):
		itTot.append(i)
		xtArray = load(folder+'xtArray_it'+str(i)+'.npy')
		ctArray = load(folder+'ctArray_it'+str(i)+'.npy')
		
		cost.append(ctArray[0])
		timeSteps.append(xtArray.shape[0]-1)

		
		plt.plot(itTot, cost, '-o',label=str(N))
		# plt.plot(itTot, timeSteps, '-*',label=str(N))

	ftocp_xPred = load(folder+'ftocp_xPred.npy')
	print("Num steps first feasible trajectory: ", ftocp_xPred.shape[0], " cost: ", cost[0])
	print("Num steps xtArray: ", xtArray.shape[0], " cost: ", cost[-1])
	# plt.legend()

	# # folder = 'data_dist_0_N_30/'
	# # folder = 'data_dist_1_N_30/'
	# folder = 'data_N_'+str(Nplot)+'/'
	# folder1 = 'data_N_'+str(Nplot1)+'/'
	
	fig = plt.figure()
	fig.set_size_inches(14, 8)
	pltCL_Color(folder,it, cl=1)
	plt.ylim(-0.01, 0.7)
	plt.show()

	# region1 = load(folder1+'region_it'+str(it)+'.npy')
	# stArray1 = load(folder1+'stArray_it'+str(it)+'.npy')
	# xtArray1 = load(folder1+'xtArray_it'+str(it)+'.npy')
	# utArray1 = load(folder1+'utArray_it'+str(it)+'.npy')
	# compTime1 = load(folder1+'compTime_it'+str(it)+'.npy')


	# ftocp_sPred = load(folder+'ftocp_sPred.npy')
	# ftocp_xPred = load(folder+'ftocp_xPred.npy')
	# ftocp_region = load(folder+'region.npy')


	# plt.figure()
	# plt.plot(compTime, '-*g')
	# plt.show()


	# plt.figure()

	# for i in range(0, ftocp_xPred.shape[0]-1):
	# 	if ftocp_region[i] == 0:
	# 		plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'or')
	# 		plt.plot(ftocp_sPred[i,0], 0, 'sk')
	# 		# print("SS left leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
	# 	elif ftocp_region[i] == 1:
	# 		plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'ob')
	# 		plt.plot(ftocp_sPred[i,1], 0, 'sk')
	# 		# print("SS right leg, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])
	# 	else:
	# 		plt.plot(ftocp_xPred[i,0], ftocp_xPred[i,1], 'oy')
	# 		plt.plot(ftocp_sPred[i,0], 0, 'sk')
	# 		plt.plot(ftocp_sPred[i,1], 0, 'sk')
	# 		# print("DS, i: ", i, "ftocp_sPred[i,0]: ", ftocp_sPred[i,:])

	# for i in range(0, xtArray.shape[0]-1):
	# 	if region[i] == 0:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], '*r')
	# 		plt.plot(stArray[i,0], 0, '*g')
	# 	elif region[i] == 1:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], '*b')
	# 		plt.plot(stArray[i,1], 0, '*g')
	# 	else:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], '*y')
	# 		plt.plot(stArray[i,0], 0, '*g')
	# 		plt.plot(stArray[i,1], 0, '*g')

	# plt.plot(xtArray[:,0], xtArray[:,1], '-k')

	# for i in range(0, xtArray1.shape[0]-1):
	# 	if region1[i] == 0:
	# 		plt.plot(xtArray1[i,0], xtArray1[i,1], 'sr')
	# 		plt.plot(stArray1[i,0], 0, 'sg')
	# 	elif region1[i] == 1:
	# 		plt.plot(xtArray1[i,0], xtArray1[i,1], 'sb')
	# 		plt.plot(stArray1[i,1], 0, 'sg')
	# 	else:
	# 		plt.plot(xtArray1[i,0], xtArray1[i,1], 'sy')
	# 		plt.plot(stArray1[i,0], 0, 'sg')
	# 		plt.plot(stArray1[i,1], 0, 'sg')

	# plt.plot(xtArray1[:,0], xtArray1[:,1], '-g')

	# pdb.set_trace()
	


	# plt.figure()
	# plt.plot(utArray[:,1], '-*r')

	# plt.figure()
	# plt.plot(utArray[:,0], '-sb')

	# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
	# fig.suptitle('Vertically stacked subplots')
	# ax1.plot(xtArray[:,0])
	# ax2.plot(xtArray[:,1])
	# ax3.plot(xtArray[:,2])
	# ax4.plot(xtArray[:,3])

	# # Compute length and variable to animate
	# lt = []
	# phaseCounter = 0
	# leg1 = []
	# leg2 = []

	# for i in range(0, xtArray.shape[0]-1):
	# 	if region[i] == 0:
	# 		leg1.append([xtArray[i,0], stArray[i,0], xtArray[i,1],   0.00])
	# 		leg2.append([xtArray[i,0], stArray[i,1], xtArray[i,1],   0.05])
	# 	elif region[i] == 1:
	# 		leg1.append([xtArray[i,0], stArray[i,0], xtArray[i,1],   0.05])
	# 		leg2.append([xtArray[i,0], stArray[i,1], xtArray[i,1],   0.00])
	# 	else:
	# 		leg1.append([xtArray[i,0], stArray[i,0], xtArray[i,1],   0.00])
	# 		leg2.append([xtArray[i,0], stArray[i,1], xtArray[i,1],   0.00])

	# variableAnimate = [np.array(leg1), np.array(leg2)]

	# # Plot
	# plt.figure()

	# for i in range(0, xtArray.shape[0]-1):
	# 	if region[i] == 0:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], 'or')
	# 		plt.plot(stArray[i,0], 0, 'sk')
	# 		print("SS left leg, i: ", i, "ftocp_sPred[i,0]: ", stArray[i,:])
	# 	elif region[i] == 1:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], 'ob')
	# 		plt.plot(stArray[i,1], 0, 'sk')
	# 		print("SS right leg, i: ", i, "ftocp_sPred[i,0]: ", stArray[i,:])
	# 	else:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], 'oy')
	# 		plt.plot(stArray[i,0], 0, 'sk')
	# 		plt.plot(stArray[i,1], 0, 'sk')
	# 		print("DS, i: ", i, "ftocp_sPred[i,0]: ", stArray[i,:])

	# plt.plot(xtArray[:,0], xtArray[:,1], '-k')

	# plt.show()

	# fig = plt.figure()
	# for i in range(0, xtArray.shape[0]-1):
	# 	if region[i] == 0:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], 'or')
	# 		plt.plot(stArray[i,0], 0, 'sk')
	# 		print("SS left leg, i: ", i, "stArray[i,0]: ", stArray[i,:])
	# 	elif region[i] == 1:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], 'ob')
	# 		plt.plot(stArray[i,1], 0, 'sk')
	# 		print("SS right leg, i: ", i, "stArray[i,0]: ", stArray[i,:])
	# 	else:
	# 		plt.plot(xtArray[i,0], xtArray[i,1], 'oy')
	# 		plt.plot(stArray[i,0], 0, 'sk')
	# 		plt.plot(stArray[i,1], 0, 'sk')
	# 		print("DS, i: ", i, "stArray[i,0]: ", stArray[i,:])

	# plt.plot(xtArray[:,0], xtArray[:,1], 'k-')

	# saveGit('closedLoop', variableAnimate, ['k', 'k'], fig)


main()
