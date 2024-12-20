import numpy as np

def get_vars(name: str, n: int, numb: bool = True):
    """Генерит символьные переменные"""
    from sympy import var, Matrix

    s = ""
    axis = ["x", "y", "z"] if n == 3 else [0, "x", "y", "z"]
    for i in range(n):
        s += f"{name}_{i} " if numb else f"{name}_{axis[i]} "

    return Matrix(var(s, real=True))

def get_func(name: str, n: int, numb: bool = True, t=None):
    """Генерит символьные функции"""
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

def numerical_and_symbolic_polymorph(trigger_var, trigger_type, trigger_out, not_trigger_out=None):
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
    return actual_decorator

def get_same_type_conversion(a):
    if isinstance(a, list):
        return list
    if isinstance(a, np.ndarray):
        return np.array
    from sympy import Matrix
    if isinstance(a, Matrix):
        return Matrix
