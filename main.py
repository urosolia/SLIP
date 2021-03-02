import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
from ftocp import FTOCP
from mpc import MPC
import datetime
from numpy import save
import pickle
import time
import os
from mpc_tracking import MPC_tracking
# Sim Options
newCost  = 0 # 1 = new min time cost, 0 = cost used to generate first feasible trajectory
dist     = 0 # 1 = disturbance active, 0 = no disturb)ance
tracking = 0 # 1 = tracking mpc, 0 = proposed method
idxIC = 2

# =============================
# Initialize system parameters
x0    = np.array([  0.00, 0.55, 0.00,0.0])   # initial condition
xref  = np.array([  10.0, 0.55, 0.00,0.0])   # initial condition

cycle = [0]*15 + [2]*4 + [1]*15 + [2]*4
region = [2]*1 + cycle*8
N  = len(region)


n  = 4
d  = 2  
dt = 0.05 
Q  = np.diag([1,10,1,1])
R  = np.diag([1,0.1])
Qf = np.diag([1000,10000,100,100])

# Initialize mpc parameters
xlb =  np.array([-100, 0.15, -5, -5])
xub =  np.array([ 100, 2.5,  5,  5])

ulb =  np.array([-10, -0.15])
uub =  np.array([ 10,  0.15])

print("cycle: ", cycle)
print("region: ", region)
print("N: ", N)

ftocp = FTOCP(N, n, d, Q, R, Qf, xlb, xub, ulb, uub, dt, xref, region)
region.append(region[-1])
# =============================
# First Feasible Trajestory
startTimer = datetime.datetime.now()
ftocp.solve(x0)
endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
print('Solver time: ', deltaTimer.total_seconds())

# Compute length
lt_l = []
lt_r = []

for i in range(0, ftocp.N):
	length = ((ftocp.xPred[i,0] - ftocp.sPred[i,0])**2 + ftocp.xPred[i,1]**2)**0.5
	lt_l.append(length)
	length = ((ftocp.xPred[i,0] - ftocp.sPred[i,1])**2 + ftocp.xPred[i,1]**2)**0.5
	lt_r.append(length)


plt.figure()
plt.plot(ftocp.xPred[0:N,0], lt_r, '-k')
plt.plot(ftocp.xPred[0:N,0], lt_l, '-b')

plt.figure()

for i in range(0, ftocp.N+1):
	if region[i] == 0:
		plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'or')
		plt.plot(ftocp.sPred[i,0], 0, 'sk')
		print("SS left leg, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])
	elif region[i] == 1:
		plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'ob')
		plt.plot(ftocp.sPred[i,1], 0, 'sk')
		print("SS right leg, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])
	else:
		plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'oy')
		plt.plot(ftocp.sPred[i,0], 0, 'sk')
		plt.plot(ftocp.sPred[i,1], 0, 'sk')
		print("DS, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])

plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '-k')

plt.show()

# =============================
# MPC
N_mpc = 30

if tracking == 1:
	folder = 'tracking_data_dist_'+str(dist)+'_N_'+str(N_mpc)+'_idxIC_'+str(idxIC)+'/'
elif newCost == 1:
	folder = 'data_N_'+str(N_mpc)+'/'
else:
	folder = 'data_dist_'+str(dist)+'_N_'+str(N_mpc)+'_idxIC_'+str(idxIC)+'/'

if not os.path.exists(folder):
    os.makedirs(folder)

save(folder+'ftocp_sPred.npy', ftocp.sPred)
save(folder+'ftocp_xPred.npy', ftocp.xPred)
save(folder+'ftocp_sPred.npy', ftocp.sPred)

pdb.set_trace()

if newCost == 1:
	Q  = 10**(-4)*np.diag([1,10,1,1])
	R  = 10**(-4)*np.diag([1,0.1])
	maxIt = 41
else:
	maxIt = 1

if tracking == 0:
	mpc = MPC(N_mpc, n, d, Q, R, Qf, xlb, xub, ulb, uub, dt, xref)
else:
	Q  = 100*np.diag([1,1,1,1])
	R  = np.diag([1,1])
	mpc = MPC_tracking(N_mpc, n, d, Q, R, Qf, xlb, xub, ulb, uub, dt)

mpc.addTraj(ftocp.xPred, ftocp.uPred, ftocp.sPred, region)

tMax = 600
pltFlag = 0

IC = []
IC.append(np.array([0.0,    0.0,   0.0,   0.0])) # 0
IC.append(np.array([0.0,   -0.03,  0.0,   0.0])) # 1
IC.append(np.array([0.005,  0.0,   0.0,   0.0])) # 2
IC.append(np.array([0.0,    0.0,   0.0,   0.2])) # 3 
IC.append(np.array([0.0,    0.0,   0.05,  0.0])) # 4 
IC.append(np.array([0.0,   -0.03,  0.03,  0.0])) # 5
IC.append(np.array([0.025,  0.0,  -0.05,  0.0])) # 6
IC.append(np.array([0.0,   -0.04,  0.00, -0.3])) # 7
IC.append(np.array([0.0,   -0.03,  0.02,  0.1])) # 8
IC.append(np.array([0.005, -0.01,  0.00, -0.1])) # 9
IC.append(np.array([0.005,  0.0,  -0.015, -0.1])) # 10


for it in range(0, maxIt):
	xt = [x0 + IC[idxIC]]
	ut = []
	st = [ftocp.sPred[0,:]]
	rt = [region[0]]
	compTime = []

	for i in range(0, tMax):
		# Solve MPC
		print("======================== Time index i: ", i, ". It: ", it)
		# if it > 0:
		# 	mpc.solve(xt[-1], st[-1])
		# 	pdb.set_trace()
		# else:
		# 	mpc.solveMultiple(xt[-1], st[-1], probToSolve = 3)

		if newCost == 0:
			mpc.solve(xt[-1], st[-1],  rt[-1])
		else:
			mpc.solveMultiple(xt[-1], st[-1],  rt[-1], probToSolve = 20)
		compTime.append(mpc.solverTime)
		
		if mpc.feasible == 0:
			pdb.set_trace()
			break

		# Propagate System (No Model Mismatch)
		if mpc.time == 103 and dist == 1: #43 for SS # 150 for DS
			# xt.append(mpc.xPred[1,:] + np.array([0.0,0.0,0.05,0.0]))
			xt.append(mpc.xPred[1,:] + np.array([0.0,0.0,0.2,0.1]))
		else:
			xt.append(mpc.xPred[1,:])
		ut.append(mpc.uPred[0,:])
		st.append(mpc.sPred[1,:])
		rt.append(mpc.region[1])

		# Shrink horizon if needed and terminate sim
		if np.linalg.norm(mpc.xPred[-1,:] - mpc.xSS[-1][-1,:] ) < 10**(-5):
			# print("mpc.xPred[-1,:]: ", mpc.xPred[-1,:], ",  mpc.xSS[-1][-1,:]: ",  mpc.xSS[-1][-1,:])	
			# print("np.linalg.norm(mpc.xPred[-1,:] - mpc.xSS[-1][-1,:] ): ", np.linalg.norm(mpc.xPred[-1,:] - mpc.xSS[-1][-1,:] ))
			if mpc.N > 2:
				mpc.N = mpc.N - 1
			else:
				xt.append(mpc.xPred[2,:])
				st.append(mpc.sPred[2,:])
				ut.append(mpc.uPred[1,:])
				rt.append(mpc.region[1])
				print("xt: ", xt[-1])
				break
		# else:
		# 	print("mpc.xPred[-1,:]: ", mpc.xPred[-1,:], ",  mpc.xSS[-1][-1,:]: ",  mpc.xSS[-1][-1,:])	
		# 	print("np.linalg.norm(mpc.xPred[-1,:] - mpc.xSS[-1][-1,:] ): ", np.linalg.norm(mpc.xPred[-1,:] - mpc.xSS[-1][-1,:] ))

		print("xt: ", xt[-1])
		# Plot predicted trajectory if flag activated
		if (pltFlag == 1):
			plt.figure()
			for i in range(0, ftocp.N):
				if region[i] == 0:
					plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'or')
					plt.plot(ftocp.sPred[i,0], 0, 'sk')
					# print("SS left leg, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])
				elif region[i] == 1:
					plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'ob')
					plt.plot(ftocp.sPred[i,1], 0, 'sk')
					# print("SS right leg, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])
				else:
					plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'oy')
					plt.plot(ftocp.sPred[i,0], 0, 'sk')
					plt.plot(ftocp.sPred[i,1], 0, 'sk')
					# print("DS, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])

			plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '-k')
			plt.plot(mpc.xPred[:,0], mpc.xPred[:,1], '-*k')

			plt.show()

	xtArray = np.array(xt)
	utArray = np.array(ut)
	stArray = np.array(st)

	save(folder+'compTime_it'+str(it)+'.npy', compTime)
	save(folder+'xtArray_it'+str(it)+'.npy', xtArray)
	save(folder+'utArray_it'+str(it)+'.npy', utArray)
	save(folder+'stArray_it'+str(it)+'.npy', stArray)
	save(folder+'ctArray_it'+str(it)+'.npy', mpc.cSS[-1])
	save(folder+'region_it'+str(it)+'.npy', rt)

	mpc.timeReset()
	mpc.addTraj(xtArray, utArray, stArray, rt)

	print(region)
	regionNew = rt
	print(regionNew)

	print("Cost diff: ", mpc.cSS[-2][0] - mpc.cSS[-1][0])

	# plt.figure()
	# plt.plot(xtArray[:,0], xtArray[:,1], '*k-')
	# plt.show()
	# pdb.set_trace()


print("========== Done with Sim")

# Compute length
plt.figure()

for i in range(0, ftocp.N):
	if region[i] == 0:
		plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'or')
		plt.plot(ftocp.sPred[i,0], 0, 'sk')
		# print("SS left leg, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])
	elif region[i] == 1:
		plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'ob')
		plt.plot(ftocp.sPred[i,1], 0, 'sk')
		# print("SS right leg, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])
	else:
		plt.plot(ftocp.xPred[i,0], ftocp.xPred[i,1], 'oy')
		plt.plot(ftocp.sPred[i,0], 0, 'sk')
		plt.plot(ftocp.sPred[i,1], 0, 'sk')
		# print("DS, i: ", i, "ftocp.sPred[i,0]: ", ftocp.sPred[i,:])

for i in range(0, xtArray.shape[0]):
	if regionNew[i] == 0:
		plt.plot(xtArray[i,0], xtArray[i,1], '*r')
		plt.plot(stArray[i,0], 0, '*g')
	elif regionNew[i] == 1:
		plt.plot(xtArray[i,0], xtArray[i,1], '*b')
		plt.plot(stArray[i,1], 0, '*g')
	else:
		plt.plot(xtArray[i,0], xtArray[i,1], '*y')
		plt.plot(stArray[i,0], 0, '*g')
		plt.plot(stArray[i,1], 0, '*g')

plt.plot(xtArray[:,0], xtArray[:,1], '-k')
plt.show()


plt.figure()

y_axis = []
x_axis = []
for it in range(0, len(mpc.cSS)):
	y_axis.append(mpc.cSS[it][0])
	x_axis.append(it)
plt.plot(x_axis, y_axis, '-ok')

y_axis = []
x_axis = []
for it in range(0, len(mpc.cSS)):
	y_axis.append(mpc.xSS[it].shape[0])
	x_axis.append(it)
plt.plot(x_axis, y_axis, '-*r')

plt.show()

save(folder+'xtArray.npy', xtArray)
save(folder+'stArray.npy', stArray)
save(folder+'region.npy', regionNew)
save(folder+'xtArray.npy', ftocp.xPred)
save(folder+'stArray.npy', ftocp.sPred)
save(folder+'region.npy', region)