# cython: language_level=3
from libcpp.string cimport string
from libcpp cimport bool
from eigency.core cimport *

ctypedef unsigned int uint

cdef extern from "prop.cpp":
    pass

cdef extern from "prop.h" namespace "propagation":
    cdef struct Channel:
        int type
        bool is_thr
        bool is_acc

        int hop
        int dim
        float delta
        float alpha
        float rra
        float rrb

    cdef cppclass PropComp:
        PropComp() except+
        #         dataset,   m,    n, seed
        void load(string, uint, uint, uint)
        float compute(uint, Channel*, Map[MatrixXf] &)
