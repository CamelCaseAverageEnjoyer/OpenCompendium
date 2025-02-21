import numpy as np
import sympy

def get_vars(name: str, n: int, numb: bool = True):
    from sympy import var, Matrix
    s = ""
    axis = ["x", "y", "z"] if n == 3 else [0, "x", "y", "z"]
    for i in range(n):
        s += f"{name}_{i} " if numb else f"{name}_{axis[i]} "

    return Matrix(var(s, real=True))

def get_func(name: str, n: int, numb: bool = True, t=None):
    from sympy import Function, Matrix

    axis = ["x", "y", "z"] if n == 3 else [0, "x", "y", "z"]

    return Matrix([Function(f"{name}_{i}" if numb else f"{name}_{axis[i]}", real=True)(t) for i in range(n)])

def sympy_norm(a):
    from sympy import sqrt
    return sqrt(a.dot(a))

def sympy_append(*args):
    from sympy import Matrix, BlockMatrix
    anw = []
    for i in args:
        anw.append(i.T)
    return Matrix(BlockMatrix(anw).T)

def sympy_mean(a):
    return sum(list(a)) / len(a)

def cross(a, b):
    return np.cross(a, b) if (isinstance(a, np.ndarray) or isinstance(a, list)) else a.cross(b)

def sin(a):
    return np.sin(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.sin(a)

def cos(a):
    return np.cos(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.cos(a)

def sqrt(a):
    return np.sqrt(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.sqrt(a)

def tan(a):
    return np.tan(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.tan(a)

def arctan(a):
    return np.arctan(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.atan(a)

def append(*args):
    a = args[0][0]
    return np.append(*args) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else \
        sympy_append(*args)

def pi(a):
    """Возвращает число π в зависимости от типа переменной 'a'
    :param a: при (int | float | np.ndarray) требует численное значение π, иначе символьное"""
    return np.pi if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.pi

def mean(a):
    return np.mean(a) if (isinstance(a[0], np.ndarray) or isinstance(a[0], int) or isinstance(a[0], float)) else sympy_mean(a)

def vstack(*args):
    a = args[0][0]
    return np.vstack(args) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else sympy.Matrix.vstack(*args)

def bmat(*args):
    try:
        a = args[0][0][0]
        return np.bmat(*args) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else \
            sympy.Matrix(sympy.BlockMatrix(*args))
    except:
        a = args[0][0]
        return np.bmat(*args) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else \
            sympy.Matrix(sympy.BlockMatrix(*args))

def norm(a):
    return np.linalg.norm(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float) or isinstance(a, list)) else sympy_norm(a)

def inv(a):
    return np.linalg.inv(a) if (isinstance(a, np.ndarray) or isinstance(a, int) or isinstance(a, float)) else a.inv()

def vec_type(a, b=None):
    if b is not None:
        return np.array(a) if (isinstance(b, np.ndarray) or isinstance(b, int) or isinstance(b, float)) else sympy.Matrix(a)
    try:
        return np.array(a) if (isinstance(a[0][0], np.ndarray) or isinstance(a[0][0], int) or isinstance(a[0][0], float)) else sympy.Matrix(a)
    except:
        return np.array(a) if (isinstance(a[0], np.ndarray) or isinstance(a[0], int) or isinstance(a[0], float)) else sympy.Matrix(a)

def quat(a):
    return np.quaternion(*a) if (isinstance(a[0], np.ndarray) or isinstance(a[0], int) or isinstance(a[0], float)) else sympy.Matrix([0, a[0], a[1], a[2]])

def dot(a, b):
    return a @ b if (isinstance(a[0], np.ndarray) or isinstance(a[0], int) or isinstance(a[0], float)) else (a.T @ b)  # [0]

def block_diag(*args):
    import scipy
    return scipy.linalg.block_diag(*args) if isinstance(args[0], np.ndarray) else \
        sympy.Matrix(sympy.BlockDiagMatrix(*args))

def zeros(dims, template):
    return np.zeros(dims) if (isinstance(template, np.ndarray) or isinstance(template, int) or isinstance(template, float)) else \
        sympy.zeros(*dims)


"""def numerical_and_symbolic_polymorph(trigger_var, trigger_type, trigger_out, not_trigger_out=None):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            # print(f"args: {args} | kwargs: {kwargs}")
            trigger = kwargs[trigger_var[1]] if trigger_var[1] in kwargs.keys() else args[trigger_var[0]]
            if isinstance(trigger, trigger_type):
                from numpy import sin, cos, sqrt, tan, arctan as atan, append, pi, mean, vstack, bmat
                from numpy.linalg import norm, inv
                out_type = trigger_out
                vec_type = np.array
                quat = lambda x: np.quaternion(*x)
                dot = lambda x, y: x @ y
            else:
                from sympy import sin, cos, sqrt, Matrix, atan, tan, pi, BlockMatrix
                norm = sympy_norm
                append = sympy_append
                mean = sympy_mean
                out_type = Matrix if not_trigger_out is None else not_trigger_out
                vec_type = Matrix
                vstack = Matrix.vstack
                bmat = lambda x: Matrix(BlockMatrix(x))
                inv = lambda x: x.inv()
                quat = lambda x: Matrix([0, x[0], x[1], x[2]])
                dot = lambda x, y: (x.T @ y)[0]
            kwargs['pi'] = pi
            kwargs['sin'] = sin
            kwargs['cos'] = cos
            kwargs['tan'] = tan
            kwargs['sqrt'] = sqrt
            kwargs['norm'] = norm
            kwargs['inv'] = inv
            kwargs['atan'] = atan
            kwargs['append'] = append
            kwargs['mean'] = mean
            kwargs['quat'] = quat
            kwargs['dot'] = dot
            kwargs['out_type'] = out_type
            kwargs['vec_type'] = vec_type
            kwargs['vstack'] = vstack
            kwargs['bmat'] = bmat

            value = func(*args, **kwargs)

            if isinstance(value, tuple):
                return tuple(out_type(i) for i in value)
            return out_type(value)
        return wrapper
    return actual_decorator"""

def get_same_type_conversion(a):
    if isinstance(a, list):
        return list
    if isinstance(a, np.ndarray):
        return np.array
    from sympy import Matrix
    if isinstance(a, Matrix):
        return Matrix
