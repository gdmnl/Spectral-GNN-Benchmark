from libc.stdlib cimport malloc, free
import numpy as np

from eigency.core cimport *
from .prop_cppext cimport PropComp, Channel

cdef class PyPropComp:
	cdef PropComp c_propcomp

	def __cinit__(self):
		self.c_propcomp = PropComp()

	def preprocess(self, np.ndarray ell, np.ndarray pll, unsigned int n, unsigned int seed):
		self.c_propcomp.preprocess(Map[VectorXi](ell), Map[VectorXi](pll), n, seed)

	def compute(self, unsigned int nchn, chns, np.ndarray feat):
		cdef:
			Channel* c_chns = <Channel*> malloc(nchn * sizeof(Channel))
			float ttime
		for i in range(nchn):
			c_chns[i].type = chns[i]['type']
			c_chns[i].is_thr = (chns[i]['type'] > 1)
			c_chns[i].is_acc = (chns[i]['type'] % 2 == 1)

			c_chns[i].hop = chns[i]['hop']
			c_chns[i].dim = chns[i]['dim']
			c_chns[i].delta = chns[i]['delta']
			c_chns[i].alpha = chns[i]['alpha']
			c_chns[i].rra = chns[i]['rra']
			c_chns[i].rrb = chns[i]['rrb']

		ttime = self.c_propcomp.compute(nchn, c_chns, Map[MatrixXf](feat))
		free(c_chns)
		return ttime
