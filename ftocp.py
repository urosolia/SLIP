from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *
import time

##### FTOCP ######
class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the ftocp program solved by the above solve method
		- model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )

	"""

	def __init__(self, N, n, d, Q, R, Qf, xlb, xub, ulb, uub, dt, xref, region ):
		# Define variables
		self.region  = region

		self.xlb = xlb
		self.xub = xub

		self.ulb = ulb
		self.uub = uub

		self.N  = N
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
		

		self.buildFTOCP()
		self.solverTime = []

	def solve(self, x0, verbose=False):
			
			# Set initial condition + state and input box constraints
			self.lbx = x0.tolist() + self.xlb.tolist()*(self.N) + self.ulb.tolist()*self.N + [0, -100] + [-100.0]*self.nl*self.N
			self.ubx = x0.tolist() + self.xub.tolist()*(self.N) + self.uub.tolist()*self.N + [0,  100] + [ 100.0]*self.nl*self.N

			# Solve nonlinear programm
			start = time.time()
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
			end = time.time()
			self.solverTime = end - start

			# Check if the solution is feasible
			if (self.solver.stats()['success']):
				print("Sucess")
				self.feasible = 1
				x = sol["x"]
				self.xPred = np.array(x[0:(self.N+1)*self.n].reshape((self.n,self.N+1))).T
				self.uPred = np.array(x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.d,self.N))).T
				self.sPred = np.array(x[((self.N+1)*self.n + self.d*self.N):((self.N+1)*self.n + self.d*self.N)+self.nl*(self.N+1)].reshape((self.nl,self.N+1))).T

				self.mpcInput = self.uPred[0][0]
				print("xPred:")
				print(self.xPred)
				print("uPred:")
				print(self.uPred)
				print("sPred:")
				print(self.sPred)
			else:
				self.xPred = np.zeros((self.N+1,self.n) )
				self.uPred = np.zeros((self.N,self.d))
				self.mpcInput = []
				self.feasible = 0
				print("Unfeasible")
			
			return self.uPred[0]

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
				print("SS, leg: ", legUsed, 'i: ', i, ", legNotUsed: ", legNotUsed)	
				X_next = self.dynamics_SS(X[n*i:n*(i+1)], U[d*i], C[i*nl + legUsed])

				# Foot constraints
				self.constraint = vertcat(self.constraint, C[(i+1)*nl + legUsed]    -  C[i*nl + legUsed]              )
				self.constraint = vertcat(self.constraint, C[(i+1)*nl + legNotUsed] - (C[i*nl + legNotUsed] + U[d*i+1]) )

			else:
				print("DS, i: ", i)	
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

		self.cost = self.cost + (X[n*self.N:n*(self.N+1)]-self.xref).T @ self.Qf @ (X[n*self.N:n*(self.N+1)]-self.xref)

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#\\, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
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
