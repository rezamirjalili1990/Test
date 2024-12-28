#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

// Function prototypes
static size_t getNextParentIndx(long long MAX, PyArrayObject *O, PyArrayObject *R, PyArrayObject *N);
static size_t getNextChildIndx(PyArrayObject *N);
static int ContinuePropagation(PyArrayObject *O, PyArrayObject *N);
static size_t TwoLabelDominance(size_t i, size_t j, PyArrayObject *R, int num_nodes, int num_resources, PyArrayObject *U);


static PyObject* initialize(PyObject* self, PyObject* args) {
    int n, n_res;
    npy_intp dims[2];
    PyArrayObject *L, *U, *R, *O, *N, *V, *P, *lb, *visit;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "ii", &n, &n_res)) {
        return NULL;
    }

    npy_intp MAX = 100000;

    // Create arrays
    dims[0] = MAX;
    L = PyArray_SimpleNew(1, dims, NPY_UINT16);
    O = PyArray_SimpleNew(1, dims, NPY_BOOL);
    N = PyArray_SimpleNew(1, dims, NPY_BOOL);
    V = PyArray_SimpleNew(1, dims, NPY_UINT16);
    lb = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    dims[1] = n;
    U = PyArray_SimpleNew(2, dims, NPY_BOOL);
    P = PyArray_SimpleNew(2, dims, NPY_UINT16);
    visit = PyArray_SimpleNew(2, dims, NPY_BOOL);

    dims[1] = n_res;
    R = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    // Initialize values
    memset(PyArray_DATA(O), 0, MAX * sizeof(npy_bool));
    memset(PyArray_DATA(L), 0, MAX * sizeof(uint16_t));
    memset(PyArray_DATA(N), 1, MAX * sizeof(npy_bool));
    memset(PyArray_DATA(U), 1, MAX * n * sizeof(npy_bool));
    memset(PyArray_DATA(lb), 0, MAX * sizeof(double));
    memset(PyArray_DATA(R), 0, MAX * n_res * sizeof(double));

    *((npy_bool*)PyArray_GETPTR1(O, 0)) = 1;
    *((npy_bool*)PyArray_GETPTR1(N, 0)) = 0;

    return Py_BuildValue("OOOOOOOOO", L, N, visit, U, P, R, O, V, lb);
}


static void propagate(
    size_t Nxt, PyArrayObject *L, PyArrayObject *N, PyArrayObject *visit, PyArrayObject *U,
    PyArrayObject *P, PyArrayObject *R, PyArrayObject *O, PyArrayObject *V,
    PyArrayObject *lb, double ub, PyArrayObject *r, PyArrayObject *r_max) {
    
    int num_resources = PyArray_DIMS(r)[2];
    int num_nodes = PyArray_DIMS(U)[1];
    size_t ch_index, i, j, k, l;
    int p_v = *(unsigned short *)PyArray_GETPTR1(V, Nxt);

    // Iterate over nodes
    for (j = 0; j < num_nodes; ++j) {
        if (*(npy_bool *)PyArray_GETPTR2(U, Nxt, j)) {
            // Get next child index
            for (ch_index = 0; ch_index < PyArray_DIMS(N)[0]; ++ch_index) {
                if (*(npy_bool *)PyArray_GETPTR1(N, ch_index)) {
                    *(npy_bool *)PyArray_GETPTR1(N, ch_index) = 0;
                    break;
                }
            }

            // Update resources
            for (k = 0; k < num_resources; ++k) {
                *(double *)PyArray_GETPTR2(R, ch_index, k) =
                    *(double *)PyArray_GETPTR3(r, p_v, j, k) +
                    *(double *)PyArray_GETPTR2(R, Nxt, k);
            }

            // Update path
            for (i = 0; i < num_nodes; ++i) {
                *(unsigned short *)PyArray_GETPTR2(P, ch_index, i) =
                    *(unsigned short *)PyArray_GETPTR2(P, Nxt, i);
            }
            *(unsigned short *)PyArray_GETPTR1(L, ch_index) =
                *(unsigned short *)PyArray_GETPTR1(L, Nxt) + 1;
            *(unsigned short *)PyArray_GETPTR2(P, ch_index,
                                               *(unsigned short *)PyArray_GETPTR1(L, ch_index)) = j;

            // Update visit
            for (i = 0; i < num_nodes; ++i) {
                *(npy_bool *)PyArray_GETPTR2(visit, ch_index, i) =
                    *(npy_bool *)PyArray_GETPTR2(visit, Nxt, i);
            }
            *(npy_bool *)PyArray_GETPTR2(visit, ch_index, j) = 1;

            // Update reachable nodes
            for (i = 0; i < num_nodes; ++i) {
                *(npy_bool *)PyArray_GETPTR2(U, ch_index, i) =
                    *(npy_bool *)PyArray_GETPTR2(U, Nxt, i);
            }
            *(npy_bool *)PyArray_GETPTR2(U, ch_index, j) = 0;

            for (k = 0; k < num_nodes; ++k) {
                if (*(npy_bool *)PyArray_GETPTR2(U, ch_index, k)) {
                    for (l = 1; l < num_resources; ++l) {
                        if (*(double *)PyArray_GETPTR3(r, j, k, l) +
                                *(double *)PyArray_GETPTR2(R, ch_index, l) >
                            *(double *)PyArray_GETPTR1(r_max, l - 1)) {
                            *(npy_bool *)PyArray_GETPTR2(U, ch_index, k) = 0;
                            break;
                        }
                    }
                }
            }

            // Check whether the new path is hitting the half-point
            npy_bool flag = 1;
            for (k = 1; k < num_resources; ++k) {
                if (*(double *)PyArray_GETPTR2(R, ch_index, k) >=
                        *(double *)PyArray_GETPTR1(r_max, k - 1) / 2 &&
                    *(unsigned short *)PyArray_GETPTR1(L, ch_index) >= num_nodes / 2) {
                    flag = 0;
                    break;
                }
            }

            *(npy_bool *)PyArray_GETPTR1(O, ch_index) = flag;
            *(unsigned short *)PyArray_GETPTR1(V, ch_index) = j;
        }
    }

    *(npy_bool *)PyArray_GETPTR1(O, Nxt) = 0;
}

// Function to get the next parent index
static PyObject *py_getNextParentIndx(PyObject *self, PyObject *args) {
    long long MAX;
    PyArrayObject *O, *R, *N;

    if (!PyArg_ParseTuple(args, "LO!O!O!", &MAX, &PyArray_Type, &O, &PyArray_Type, &R, &PyArray_Type, &N)) {
        return NULL;
    }

    size_t result = getNextParentIndx(MAX, O, R, N);
    return PyLong_FromSize_t(result);
}

// Function to get the next child index
static PyObject *py_getNextChildIndx(PyObject *self, PyObject *args) {
    PyArrayObject *N;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &N)) {
        return NULL;
    }

    size_t result = getNextChildIndx(N);
    return PyLong_FromSize_t(result);
}

// Function to check if propagation should continue
static PyObject *py_ContinuePropagation(PyObject *self, PyObject *args) {
    PyArrayObject *O, *N;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &O, &PyArray_Type, &N)) {
        return NULL;
    }

    int result = ContinuePropagation(O, N);
    return PyBool_FromLong(result);
}

// Function for two-label dominance
static PyObject *py_TwoLabelDominance(PyObject *self, PyObject *args) {
    size_t i, j;
    PyArrayObject *R, *U;
    int num_nodes, num_resources;

    if (!PyArg_ParseTuple(args, "kkO!iiO!", &i, &j, &PyArray_Type, &R, &num_nodes, &num_resources, &PyArray_Type, &U)) {
        return NULL;
    }

    size_t result = TwoLabelDominance(i, j, R, num_nodes, num_resources, U);
    return PyLong_FromSize_t(result);
}

// Implementation of functions
static size_t getNextParentIndx(long long MAX, PyArrayObject *O, PyArrayObject *R, PyArrayObject *N) {
    double min_cost = 1000.0;
    size_t min_index = -1;
    double *r_row;
    npy_bool *o_ptr, *n_ptr;

    for (size_t i = 0; i < MAX; ++i) {
        o_ptr = (npy_bool *)PyArray_GETPTR1(O, i);
        n_ptr = (npy_bool *)PyArray_GETPTR1(N, i);
        r_row = (double *)PyArray_GETPTR2(R, i, 0);

        if (*o_ptr && !(*n_ptr) && min_cost > *r_row) {
            min_index = i;
            min_cost = *r_row;
        }
    }
    return min_index;
}

static size_t getNextChildIndx(PyArrayObject *N) {
    npy_bool *n_ptr;

    for (size_t i = 0; i < PyArray_SIZE(N); ++i) {
        n_ptr = (npy_bool *)PyArray_GETPTR1(N, i);
        if (*n_ptr) {
            return i;
        }
    }
    return (size_t)-1;  // Return -1 if no index found
}

static int ContinuePropagation(PyArrayObject *O, PyArrayObject *N) {
    npy_bool *o_ptr, *n_ptr;

    for (size_t i = 0; i < PyArray_SIZE(O); ++i) {
        o_ptr = (npy_bool *)PyArray_GETPTR1(O, i);
        n_ptr = (npy_bool *)PyArray_GETPTR1(N, i);

        if (*o_ptr && !(*n_ptr)) {
            return 1;  // True
        }
    }
    return 0;  // False
}

static size_t TwoLabelDominance(size_t i, size_t j, PyArrayObject *R, int num_nodes, int num_resources, PyArrayObject *U) {
    size_t k;
    int flag_i = 0, flag_j = 0;

    for (k = 0; k < num_resources; ++k) {
        double *r_i_k = (double *)PyArray_GETPTR2(R, i, k);
        double *r_j_k = (double *)PyArray_GETPTR2(R, j, k);

        if (*r_i_k <= *r_j_k) flag_i++;
        if (*r_i_k >= *r_j_k) flag_j++;
    }

    for (k = 0; k < num_nodes; ++k) {
        npy_bool *u_i_k = (npy_bool *)PyArray_GETPTR2(U, i, k);
        npy_bool *u_j_k = (npy_bool *)PyArray_GETPTR2(U, j, k);

        if (*u_i_k >= *u_j_k) flag_i++;
        if (*u_i_k <= *u_j_k) flag_j++;
    }

    if (flag_i == num_resources + num_nodes) return i;
    if (flag_j == num_resources + num_nodes) return j;

    return (size_t)-1;  // Dominance not established
}

// Module method definitions
static PyMethodDef EspprcMethods[] = {
    {"initialize", initialize, METH_NOARGS, "Initialize NumPy API."},
    {"getNextParentIndx", py_getNextParentIndx, METH_VARARGS, "Get the next parent index."},
    {"getNextChildIndx", py_getNextChildIndx, METH_VARARGS, "Get the next child index."},
    {"ContinuePropagation", py_ContinuePropagation, METH_VARARGS, "Check if propagation should continue."},
    {"TwoLabelDominance", py_TwoLabelDominance, METH_VARARGS, "Check two-label dominance."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef espprc_module = {
    PyModuleDef_HEAD_INIT,
    "espprc",
    "ESPPRC C extension module using NumPy C-API.",
    -1,
    EspprcMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_espprc(void) {
    import_array();  // Initialize NumPy API
    return PyModule_Create(&espprc_module);
}
