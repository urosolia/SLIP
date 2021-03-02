from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *
import time

##### MPC ######
class MPC(object):
	""" Model Predictive Controller (MPC)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the ftocp program solved by the above solve method
		- model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )

	"""

	def __init__(self, N, n, d, Q, R, Qf, xlb, xub, ulb, uub, dt, xref):
		# Define variables
		self.xlb = xlb
		self.xub = xub

		self.ulb = ulb
		self.uub = uub

		self.Ninit  = N
		self.n  = n
		self.nl = 2
		self.d  = d
		self.Q = Q
		self.Qf = Qf
		self.R = R

		self.xref = xref

		self.dt = dt

		self.l  = 0.7
		self.m  = 1
		self.g = 9.81
		self.k0 = self.m*self.g/0.15
		
		self.Sslack = 10**8 * np.diag([1.0,1.0])
		self.Qslack = 10**8 * np.diag([1.0,1.0,1.0,1.0])

		self.xf = np.zeros(n)
		self.solverTime = []

		# Define Terminal Constraint Set
		self.xSS = []
		self.uSS = []
		self.sSS = []
		self.rSS = []
		self.cSS = []

		self.timeReset()
	
	def timeReset(self):
		self.time = 0
		self.N = self.Ninit
		self.idxSS = self.N

	def addTraj(self, x_cl, u_cl, s_cl, region):
		self.xSS.append(x_cl)
		self.uSS.append(u_cl)
		self.rSS.append(region)
		self.sSS.append(s_cl)
		self.cSS.append(self.computeCost(x_cl, u_cl))

	def computeCost(self, x, u):
		# Compute the csot of the roll-out (sum realized cost over closed-loop trajectory)
		for i in range(0, x.shape[0]):
			idx = x.shape[0] - 1 - i 
			xt = x[idx,:]
			if i == 0:
				c = [np.dot(xt, np.dot(self.Q, xt))]
			else:
				ut = u[idx,:]
				c.append(c[-1] + np.dot(xt, np.dot(self.Q, xt)) + np.dot(ut, np.dot(self.R, ut)) + 1)

		costVector = np.array([np.flip(c)]).T

		return costVector

	def solveMultiple(self, x0, s0, r0, verbose=False, probToSolve = 1):
		self.solverTime = 0

		xPred = []
		uPred = []
		sPred = []
		cPred = []
		idxSS = []
		feas  = []
		region = []

		if self.time > 0:
			curr_xPred = self.xPred
			curr_uPred = self.uPred

		for i in range(0, probToSolve):
			if self.time > 0:
				self.xPred = curr_xPred
				self.uPred = curr_uPred
			
			self.buildTermComp(r0, i)
			self.solveFTOCP(x0, s0)

			xPred.append(self.xPred)
			uPred.append(self.uPred)
			sPred.append(self.sPred)
			cPred.append(self.cPred)
			idxSS.append(self.idxToUse)
			feas.append(self.feasible)
			region.append(self.region)

			if i > 0:
				if cPred[i-1] < cPred[i]+0.0001:
					break
			# if self.cPred > 10**8:
			# 	pdb.set_trace()

		print("List of costs: ", cPred)
		idxOpt = np.argmin(cPred)

		if cPred[idxOpt] > 10**3:
			pdb.set_trace()

		# if idxOpt > 0:
		# 	pdb.set_trace()

		if np.max(feas) == 0:
			print("all not feasible")
			pdb.set_trace()

		self.xPred    = xPred[idxOpt]
		self.uPred    = uPred[idxOpt]
		self.sPred    = sPred[idxOpt]
		self.cPred    = cPred[idxOpt]
		self.region   = region[idxOpt]
		self.feasible = feas[idxOpt]

		self.time += 1
		self.idxSS = int(np.min([idxSS[idxOpt] + 1, self.xSS[-1].shape[0]-1]))
		# print("xPred:")
		# print(self.xPred)
		# print("uPred:")
		# print(self.uPred)
		# print("sPred:")
		# print(self.sPred)

		
	def solve(self, x0, s0, r0, verbose=False):
		self.solverTime = 0

		self.buildTermComp(r0)
		self.solveFTOCP(x0, s0, verbose=verbose)

		self.time += 1
		self.idxSS = int(np.min([self.idxSS + 1, self.xSS[-1].shape[0]-1]))
		
		print("Done Solving MPC")

	def buildTermComp(self, r0, deltaTime = 0):
		# Select Terminal Constraint
		self.idxToUse = np.min([self.idxSS + deltaTime, self.xSS[-1].shape[0]-1]) 

		self.xf = self.xSS[-1][self.idxToUse, :]
		self.uf = self.uSS[-1][self.idxToUse-1, :]
		self.cf = self.cSS[-1][self.idxToUse, :]

		self.sf = self.sSS[-1][self.idxToUse, :]
		self.region = self.rSS[-1][(self.idxToUse-self.N):(self.idxToUse)]
		self.region[0] = r0

		print("xf: ", self.xf)
		print("self.sf: ", self.sf)

		self.buildFTOCP()
		




	def solveFTOCP(self, x0, s0, warmStart=True, verbose=False):
		# print("s0: ", s0)

		# Set initial condition + state and input box constraints
		self.lbx = x0.tolist() + self.xlb.tolist()*(self.N) + self.ulb.tolist()*self.N + s0.tolist() + [-100.0]*self.nl*self.N
		self.ubx = x0.tolist() + self.xub.tolist()*(self.N) + self.uub.tolist()*self.N + s0.tolist() + [ 100.0]*self.nl*self.N

		# Pick warm start
		if self.time == 0:
			xGuess = self.xSS[-1][self.time:(self.time+self.N+1), :].reshape(self.n*(self.N+1))
			uGuess = self.uSS[-1][self.time:(self.time+self.N+0), :].reshape(self.d*(self.N+0))
			sGuess = self.sSS[-1][self.time:(self.time+self.N+1), :].reshape(self.nl*(self.N+1))
		else:
			# xGuess = self.xPred[0:(self.N+1), :].reshape(self.n*(self.N+1))
			# uGuess = self.uPred[0:(self.N+0), :].reshape(self.d*(self.N+0))
			xGuess = np.append(self.xPred[1:(self.N+1), :].reshape(self.n*(self.N-0)), self.xf)
			uGuess = np.append(self.uPred[1:(self.N+0), :].reshape(self.d*(self.N-1)), self.uf)
			sGuess = np.append(self.sPred[1:(self.N+1), :].reshape(self.nl*(self.N-0)), self.sf)
		
		zGuess = np.append(np.append(xGuess,uGuess), sGuess)

		# Solve nonlinear programm
		start = time.time()		
		if warmStart == False:
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
		else:
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0=zGuess.tolist())
		end = time.time()
		self.solverTime += end - start

		# Check if the solution is feasible
		if (self.solver.stats()['success']):

			self.feasible = 1
			x = sol["x"]
			self.cPred = sol['f']
			self.xPred = np.array(x[0:(self.N+1)*self.n].reshape((self.n,self.N+1))).T
			self.uPred = np.array(x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.d,self.N))).T
			self.sPred = np.array(x[((self.N+1)*self.n + self.d*self.N):((self.N+1)*self.n + self.d*self.N)+self.nl*(self.N+1)].reshape((self.nl,self.N+1))).T

			self.mpcInput = self.uPred[0][0]
			# print("xPred:")
			# print(self.xPred)
			# print("uPred:")
			# print(self.uPred)
			# print("sPred:")
			# print(self.sPred)
			# print("Term state: ", self.xPred[-1,:])
			# print("Sucess")
			
		else:
			self.cPred = 10**10
			self.xPred = np.zeros((self.N+1,self.n) )
			self.uPred = np.zeros((self.N,self.d))
			self.mpcInput = []
			self.feasible = 0
			print("Unfeasible")
			# pdb.set_trace()
			# sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
			# if (self.solver.stats()['success']):
			# 	print("Sucess")
			# 	self.feasible = 1
			# 	x = sol["x"]
			# 	self.xPred = np.array(x[0:(self.N+1)*self.n].reshape((self.n,self.N+1))).T
			# 	self.uPred = np.array(x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.d,self.N))).T
			# 	self.sPred = np.array(x[((self.N+1)*self.n + self.d*self.N):((self.N+1)*self.n + self.d*self.N)+self.s])

			# 	self.mpcInput = self.uPred[0][0]
			# 	print("xPred:")
			# 	print(self.xPred)
			# 	print("uPred:")
			# 	print(self.uPred)
			# 	print("sPred:")
			# 	print(self.sPred)
			# else:
			# 	print("Also new problem not feasible")

		# print(self.xSS[-1][self.time:self.time+self.N+1,:])
		# print(self.uSS[-1][self.time:self.time+self.N,:])

	def buildFTOCP(self):
		# Define variables
		n  = self.n
		d  = self.d
		nl  = self.nl

		# Define variables
		X      = SX.sym('X', n*(self.N+1));
		U      = SX.sym('U', d*self.N);
		C      = SX.sym('C', nl*(self.N+1));

		# Define dynamic constraints
		self.constraint = []
		for i in range(0, self.N):
			if self.region[i] < 2:
				legUsed    = self.region[i]
				legNotUsed = 1 - legUsed
				# print("SS, leg: ", legUsed, 'i: ', i, ", legNotUsed: ", legNotUsed)	
				X_next = self.dynamics_SS(X[n*i:n*(i+1)], U[d*i], C[i*nl + legUsed])

				# Foot constraints
				self.constraint = vertcat(self.constraint, C[(i+1)*nl + legUsed]    -  C[i*nl + legUsed]              )
				self.constraint = vertcat(self.constraint, C[(i+1)*nl + legNotUsed] - (C[i*nl + legNotUsed] + U[d*i+1]) )

			else:
				# print("DS, i: ", i)	
				X_next = self.dynamics_DS(X[n*i:n*(i+1)], U[d*i], C[i*nl:(i+1)*nl])

				# Foot constraints
				for j in range(0,2):
					self.constraint = vertcat(self.constraint, C[(i+1)*nl + j] - C[i*nl + j] )
	
			# Dyanmic update
			for j in range(0, self.n):
				self.constraint = vertcat(self.constraint, X_next[j] - X[n*(i+1)+j] )


		# Constraints on length
		lbg_leg = []
		ubg_leg = []
		for i in range(0, self.N):

			if self.region[i] == 0:
				self.constraint = vertcat(self.constraint, (X[n*i + 0] - C[i*nl+0])**2 + X[n*i + 1]**2 )
				# Leg 0 length <= 1
				lbg_leg.append(0)
				ubg_leg.append(self.l**2)
			elif self.region[i] == 1:
				self.constraint = vertcat(self.constraint, (X[n*i + 0] - C[i*nl+1])**2 + X[n*i + 1]**2 )
				# Leg 1 length <= 1
				lbg_leg.append(0)
				ubg_leg.append(self.l**2)
			else:
				self.constraint = vertcat(self.constraint, (X[n*i + 0] - C[i*nl+0])**2 + X[n*i + 1]**2 )
				self.constraint = vertcat(self.constraint, (X[n*i + 0] - C[i*nl+1])**2 + X[n*i + 1]**2 )
				# Leg 0 length <= 1
				lbg_leg.append(0)
				ubg_leg.append(self.l**2)
				# Leg 1 length <= 1
				lbg_leg.append(0)
				ubg_leg.append(self.l**2)

		# Defining Cost
		self.cost = 0
		for i in range(0, self.N):
			self.cost = self.cost + (X[n*i:n*(i+1)]-self.xref).T @ self.Q @ (X[n*i:n*(i+1)]-self.xref)
			self.cost = self.cost + U[d*i:d*(i+1)].T @ self.R @ U[d*i:d*(i+1)]

		self.cost = self.cost + (X[n*self.N:n*(self.N+1)]-self.xref).T @ self.Q @ (X[n*self.N:n*(self.N+1)]-self.xref)

		# Terminal Constraints
		self.cost = self.cost + (X[n*self.N:n*(self.N+1)]  -self.xf).T @ self.Qslack @ (X[n*self.N:n*(self.N+1)]-self.xf)
		self.cost = self.cost + (C[nl*self.N:nl*(self.N+1)]-self.sf).T @ self.Sslack @ (C[nl*self.N:nl*(self.N+1)]-self.sf)
		self.cost = self.cost + self.cf

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0, "ipopt.max_cpu_time":0.5}#\\, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U,C), 'f':self.cost, 'g':self.constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force n*N state dynamics
		self.lbg_dyanmics = [0]*((n+nl)*self.N) + lbg_leg
		self.ubg_dyanmics = [0]*((n+nl)*self.N) + ubg_leg

	def dynamics_SS(self, x, u, c):
		theta = np.arctan( (x[0] - c)/ x[1] )
		leng  = np.sqrt( (x[0] - c)**2 + x[1]**2 )

		# state x = [x,y, vx, vy]
		x_next  = x[0] + self.dt * x[2]
		y_next  = x[1] + self.dt * x[3]
		vx_next = x[2] + self.dt * ( sin(theta)*((u+self.k0)*(self.l - leng) ) )
		vy_next = x[3] + self.dt * (-self.m*self.g + cos(theta)*((u+self.k0)*(self.l - leng) ) ) #+ cos(theta)*u[1] )

		state_next = [x_next, y_next, vx_next, vy_next]

		return state_next

	def dynamics_DS(self, x, u, c):
		theta = []
		leng  = []

		for i in [0, 1]:
			theta.append(np.arctan( (x[0] - c[i])/ x[1] ))
			leng.append(np.sqrt( (x[0] - c[i])**2 + x[1]**2 ))

		# state x = [x,y, vx, vy]
		x_next  = x[0] + self.dt * x[2]
		y_next  = x[1] + self.dt * x[3]
		vx_next = x[2] + self.dt * (sin(theta[0])*((u[0]+self.k0)*(self.l - leng[0]) ) + sin(theta[1])*((u+self.k0)*(self.l - leng[1]) ))
		vy_next = x[3] + self.dt * (cos(theta[0])*((u[0]+self.k0)*(self.l - leng[0]) ) + cos(theta[1])*((u+self.k0)*(self.l - leng[1]) - self.m*self.g)) 

		state_next = [x_next, y_next, vx_next, vy_next]

		return state_next
