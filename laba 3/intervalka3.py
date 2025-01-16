import copy
import struct
import numpy as np
np.float_ = np.float64
import intvalpy as ip
import matplotlib.pyplot as plt
from functools import cmp_to_key

ip.precision.extendedPrecisionQ = False

def union_intervals(x, y):
    return ip.Interval(min(x.a, y.a), max(x.b, y.b))

def are_adjusted_to_each_other(x, y):
  return x.b == y.a or y.b == x.a

def are_intersected(x, y):
  sup = y.a if x.a < y.a else x.a
  inf = x.b if x.b < y.b else y.b
  return sup - inf <= 1e-15

def merge_intervals(x, y):
  return ip.Interval(min(x.a, y.a), max(x.b, y.b))

def mode(x):
  if len(x) == 0:
    return []

  edges = sorted({x_i.a for x_i in x}.union({x_i.b for x_i in x}))
  z = [ip.Interval(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
  mu = [sum(1 for x_i in x if z_i in x_i) for z_i in z]

  max_mu = max(mu)
  K = [index for index, element in enumerate(mu) if element == max_mu]

  m = [z[k] for k in K]
  merged_m = []

  current_interval = m[0]

  for next_interval in m[1:]:
    if are_intersected(current_interval, next_interval) or are_adjusted_to_each_other(current_interval, next_interval):
      current_interval = merge_intervals(current_interval, next_interval)
    else:
      merged_m.append(current_interval)
      current_interval = next_interval

  merged_m.append(current_interval)

  return merged_m

def med_k(x):
  starts = [float(interval.a) for interval in x]
  ends = [float(interval.b) for interval in x]
  return ip.Interval(np.median(starts), np.median(ends))

def med_p(x):
  x = sorted(x, key=cmp_to_key(lambda x, y: (x.a + x.b) / 2 - (y.a + y.b) / 2))

  mid_index = len(x) // 2

  if len(x) % 2 == 0:
    return (x[mid_index - 1] + x[mid_index]) / 2

  return x[mid_index]


def jaccard_index(*args):
  if len(args) == 1:
    x = args[0]
    left_edges = [interval.a for interval in x]
    right_edges = [interval.b for interval in x]

    return (min(right_edges) - max(left_edges)) / (max(right_edges) - min(left_edges))
  elif len(args) == 2:
    x = args[0]
    y = args[1]

    if isinstance(x, ip.ClassicalArithmetic) and isinstance(y, ip.ClassicalArithmetic):
      return (min(x.b, y.b) - max(x.a, y.a)) / (max(x.b, y.b) - min(x.a, y.a))
    else:
      results = []

      for x_i, y_i in zip(x, y):
        result = (min(x_i.b, y_i.b) - max(x_i.a, y_i.a)) / (max(x_i.b, y_i.b) - min(x_i.a, y_i.a))
        results.append(result)

      return np.array(results)
  else:
    raise ValueError("Wrong number of arguments")


def read_bin_file_with_numpy(file_path):
    with open(file_path, 'rb') as f:
        header_data = f.read(256)
        side, mode_, frame_count = struct.unpack('<BBH', header_data[:4])

        frames = []
        point_dtype = np.dtype('<8H')

        for _ in range(frame_count):
            frame_header_data = f.read(16)
            stop_point, timestamp = struct.unpack('<HL', frame_header_data[:6])
            frame_data = np.frombuffer(f.read(1024 * 16), dtype=point_dtype)
            frames.append(frame_data)
        print("Complete load data")
        return np.array(frames)


def get_avg(data):
    avg = [[0]*8]*1024
    for i in range(len(data)): # 100
        avg = np.add(avg, data[i])
    return np.divide(avg, len(data))


def scalar_to_interval(x, rad):
    return ip.Interval(x - rad, x + rad)


def argmaxF(f, a, b, eps):
    lmbd = a + (3 - 5 ** 0.5) * (b - a)/2
    mu = b - (3 - 5 ** 0.5) * (b - a) / 2
    f_lambda = f(lmbd)
    f_mu = f(mu)

    while 1:
        if f_lambda <= f_mu:
            a = lmbd
            if eps > b - a:
                break
            lmbd = mu
            f_lambda = f_mu
            mu = b - (3 - 5 ** 0.5) * (b - a) / 2
            f_mu = f(mu)
        else:
            b = mu
            if eps > b - a:
                break
            mu = lmbd
            f_mu = f_lambda
            lmbd = a + (3 - 5 ** 0.5) * (b - a)/2
            f_lambda = f(lmbd)

        # print(a)
        # print(b)

    return (a+b) / 2


def func_a(a):
    return np.mean(jaccard_index(X + a, Y))


def func_t(t):
    return np.mean(jaccard_index(X * t, Y))


def func_a_mode(a):
    return np.mean(jaccard_index(mode(X + a), mode(Y)))


def func_t_mode(t):
    return np.mean(jaccard_index(mode(X * t), mode(Y)))


def func_a_med_p(a):
    return np.mean(jaccard_index(med_p(X + a), med_p(Y)))


def func_t_med_p(t):
    return np.mean(jaccard_index(med_p(X * t), med_p(Y)))


def func_a_med_k(a):
    return np.mean(jaccard_index(med_k(X + a), med_k(Y)))


def func_t_med_k(t):
    return np.mean(jaccard_index(med_k(X * t), med_k(Y)))


def draw_func(f, a, b, parametr: str, func=""):
    X_linsp = np.linspace(a, b, 100)
    y = np.array([f(x) for x in X_linsp])
    plt.plot(X_linsp, y, )

    plt.xlabel(f"{parametr}")
    plt.ylabel(f"Ji({parametr}, {func}(X), {func}(Y))")
    plt.title("Jaccard Index")
    plt.savefig(f"Jaccadrd-{parametr}-{func}")
    plt.show()


scalar_to_interval_vec = np.vectorize(scalar_to_interval)

x_data = read_bin_file_with_numpy('-0.205_lvl_side_a_fast_data.bin')

y_data = read_bin_file_with_numpy('0.225_lvl_side_a_fast_data.bin')

x_data = get_avg(x_data)
y_data = get_avg(y_data)

x_voltage = x_data / 16384.0 - 0.5
y_voltage = y_data / 16384.0 - 0.5

rad = 2 ** (-14)

X = scalar_to_interval_vec(x_voltage, rad).flatten()
Y = scalar_to_interval_vec(y_voltage, rad).flatten()

# # Функционал = Ji(const, X, Y)
draw_func(func_a, 0, 1, "a")
a_f = argmaxF(func_a, 0, 1, 1e-3)
print(a_f, func_a(a_f))
draw_func(func_t, -4, 0, "t")
t_f = argmaxF(func_t, -4, 0, 1e-3)
print(t_f, func_t(t_f))

# # Функционал = Ji(const,mode(X), mode(Y))
draw_func(func_a_mode, 0, 1, "a", "mode")
a_f_mode = argmaxF(func_a_mode, 0, 1, 1e-3)
print(a_f_mode, func_a_mode(a_f_mode))
draw_func(func_t_mode, -4, 0, "a", "mode")
t_f_mode = argmaxF(func_t_mode, -4, 0, 1e-3)
print(t_f_mode, func_t_mode(t_f_mode))

# # Функционал = Ji(const,med_р(X), med_р(Y))
draw_func(func_a_med_p, 0, 1, "a", "med_p")
a_f_med_p = argmaxF(func_a_med_p, 0, 1, 1e-3)
print(a_f_med_p, func_a_med_p(a_f_med_p))
draw_func(func_t_med_p, -4, 0, "t", "med_p")
t_f_med_p = argmaxF(func_t_med_p, -4, 0, 1e-3)
print(t_f_med_p, func_t_med_p(t_f_med_p))

# # Функционал = Ji(const,med_K(X), med_K(Y))
draw_func(func_a_med_k, 0, 1, "a", "med_K")
a_f_med_k = argmaxF(func_a_med_k, 0, 1, 1e-3)
print(a_f_med_k, func_a_med_k(a_f_med_k))
draw_func(func_t_med_k, -4, 0, "t", "med_K")
t_f_med_k = argmaxF(func_t_med_k, -4, 0, 1e-3)
print(t_f_med_k, func_t_med_k(t_f_med_k))
