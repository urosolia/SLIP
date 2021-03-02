import numpy as np
import pdb
from scipy.spatial import ConvexHull


class system(object):
	"""docstring for system"""
	def __init__(self, A, B, w_inf, x0):
		self.A = A
		self.B = B
		self.w_inf = w_inf
		self.x = [x0]
		self.u = []
		self.w = []

		self.w_v = []
		self.w_v.append(w_inf*np.array([ 1 ,1]))
		self.w_v.append(w_inf*np.array([ 1,-1]))
		self.w_v.append(w_inf*np.array([-1, 1]))
		self.w_v.append(w_inf*np.array([-1,-1]))
		self.computeRobutInvariant()
		
	def applyInput(self, ut):
		self.u.append(ut)
		self.w.append(np.array([0,0]))
		xnext = np.dot(self.A,self.x[-1]) + np.dot(self.B,self.u[-1]) + self.w[-1]
		self.x.append(xnext)

	def computeRobutInvariant(self):
		self.O_v = [np.array([0,0])]
		print("Compute robust invariant")
		# TO DO:
		# - add check for convergence
		# - add check for input and state constraint satifaction
		for i in range(0,20):
			self.O_v = self.MinkowskiSum(self.O_v, self.w_v)

	def MinkowskiSum(self, setA, setB):
		vertices = []
		for v1 in setA:
			for v2 in setB:
				vertices.append(np.dot(self.A,v1) + v2)
		cvxHull = ConvexHull(vertices)
		verticesOut = []
		for idx in cvxHull.vertices:
			verticesOut.append(vertices[idx])
		return verticesOut
