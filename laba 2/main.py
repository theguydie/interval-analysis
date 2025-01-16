import copy
import numpy as np
np.float_ = np.float64
import intvalpy as ip
import matplotlib.pyplot as plt
from tol import tolsolvty

ip.precision.extendedPrecisionQ = False


def emptinessTol(A, b):
    maxTol = ip.linear.Tol.maximize(A, b)
    if maxTol[1] < 0:
        return True, maxTol[0], maxTol[1]
    else:
        return False, maxTol[0], maxTol[1]


def b_correction(A, b, step=5):
    mid = ip.mid(b)
    new_rad = ip.rad(b) + step
    new_b = [[mid[i] - new_rad[i], mid[i] + new_rad[i]] for i in range(len(mid))]
    return ip.Interval(new_b)



def A_correction(A, b):
    max_tol = ip.linear.Tol.maximize(A, b)
    lower_bound = abs(max_tol[1]) / (abs(max_tol[0][0]) + abs(max_tol[0][1]))
    rad_A = ip.rad(A)
    upper_bound = rad_A[0][0]
    for a_i in rad_A:
        for a_ij in a_i:
            if a_ij < upper_bound:
                upper_bound = a_ij
    e = (lower_bound + upper_bound) / 2
    corrected_A = []
    for i in range(len(A)):
        A_i = []
        for j in range(len(A[0])):
            A_i.append([A[i][j]._a + e, A[i][j]._b - e])
        corrected_A.append(A_i)
    return ip.Interval(corrected_A)


def Ab_correction(A, b):
    emptiness, max_x, max_Tol = emptinessTol(A, b)
    new_A = copy.deepcopy(A)
    new_b = copy.deepcopy(b)

    while emptiness:
        new_A = A_correction(new_A, new_b)
        emptiness, max_x, max_Tol = emptinessTol(new_A, new_b)
        if not emptiness:
            break

        new_b = b_correction(new_A, new_b)
        emptiness, max_x, max_Tol = emptinessTol(new_A, new_b)

    return new_A, new_b


def draw_Tol(A, b, max_x, max_Tol, name: str, ax):
    x_1_, x_2_ = np.mgrid[max_x[0] - 2:max_x[0] + 2:100j, max_x[1] - 2:max_x[1] + 2:100j]

    list_x_1 = np.linspace(max_x[0] - 2, max_x[0] + 2, 100)
    list_x_2 = np.linspace(max_x[1] - 2, max_x[1] + 2, 100)

    list_tol = []
    for x_1 in list_x_1:
        mas_tol = []
        for x_2 in list_x_2:
            x = [x_1, x_2]
            tol = []
            for i in range(len(b)):
                sum_ = sum([A[i][j] * x[j] for j in range(len(x))])
                tol.append(ip.rad(b[i]) - ip.mag(ip.mid(b[i]) - sum_))

            mas_tol.append(min(tol))

        list_tol.append(np.array(mas_tol))

    ax.plot_surface(x_1_, x_2_, np.array(list_tol), cmap='viridis', edgecolor='yellow')
    ax.set_title(f"Tol: {name[7:]}")
    ax.scatter(max_x[0], max_x[1], max_Tol, color='red')


A1 = ip.Interval([
    [[0.65, 1.25], [0.7, 1.3]],
    [[0.75, 1.35], [0.7, 1.3]]
])
b1 = ip.Interval([[2.75, 3.15],
                 [2.85, 3.25]])


A2 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]],
    [[0.8, 1.4], [0.70, 1.3]]
])

b2 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.3]
    ])

A3 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]],
    [[0.8, 1.4], [0.70, 1.3]],
    [[-0.3, 0.3], [0.70, 1.3]]
])

b3 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.3],
    [1.8, 2.2],
    ])

As = [A1, A2, A3]
bs = [b1, b2, b3]


def run(correction=None):
    match correction:
        case "Ab":
            print("____Ab-correction____")
            fig = plt.figure(figsize=(18, 6))
            for i in range(len(As)):
                A_ = As[i]
                b_ = bs[i]

                A_, b_ = Ab_correction(A_, b_)
                ip.IntLinIncR2(A_, b_, consistency='tol', title=f"Ab-correction {i + 1}", save=True, size=(8, 8), show=True)
                plt.scatter(1, 2)
                # plt.show()

                emptiness_, maxX, maxTol = emptinessTol(A_, b_)
                print(ip.sup(b_).reshape(-1, 1))
                tolmax, argmax, envs, ccode = tolsolvty(infA=ip.inf(A_), supA=ip.sup(A_),
                                                        infb=ip.inf(b_).reshape(-1, 1), supb=ip.sup(b_).reshape(-1, 1))

                print(tolmax)
                print(argmax)
                print(envs)
                print(emptiness_, maxX, maxTol)
                draw_Tol(A_, b_, maxX, maxTol, f"Graphs/Ab-correction {i + 1}", fig.add_subplot(1, 3, i + 1, projection='3d'))
            fig.savefig(f"Graphs/Ab-corrections")
        case "A":
            print("____A-correction____")
            fig = plt.figure(figsize=(18, 6))
            for i in range(len(As)):
                A_ = As[i]
                b_ = bs[i]

                A_ = A_correction(A_, b_)
                print(A_)
                print(emptinessTol(A_, b_))
                ip.IntLinIncR2(A_, b_, consistency='tol', title=f"A-correction {i + 1}", save=True, size=(8, 8), show=True)
                plt.scatter(1, 2)
                # plt.show()

                emptiness_, maxX, maxTol = emptinessTol(A_, b_)
                print(emptiness_)

                print(ip.sup(b_).reshape(-1, 1))
                tolmax, argmax, envs, ccode = tolsolvty(infA=ip.inf(A_), supA=ip.sup(A_),
                                                        infb=ip.inf(b_).reshape(-1, 1), supb=ip.sup(b_).reshape(-1, 1))
                print(tolmax)
                print(argmax)
                print(envs)

                draw_Tol(A_, b_, maxX, maxTol, f"Graphs/A-correction {i + 1}", fig.add_subplot(1, 3, i + 1, projection='3d'))
            fig.savefig(f"Graphs/A-corrections")
        case "b":
            print("____b-correction____")
            fig = plt.figure(figsize=(18, 6))
            for i in range(len(As)):
                A_ = As[i]
                b_ = bs[i]

                b_ = b_correction(A_, b_)
                print(b_)
                print(emptinessTol(A_, b_))

                figure = plt.figure(figsize=(18, 6))
                ip.IntLinIncR2(A_, b_, consistency='tol', title=f"b-correction {i + 1}", save=True, size=(8, 8), show=True)
                plt.scatter(1, 2)
                # plt.show()

                emptiness_, maxX, maxTol = emptinessTol(A_, b_)
                print(emptiness_)

                print(ip.sup(b_).reshape(-1, 1))
                tolmax, argmax, envs, ccode = tolsolvty(infA=ip.inf(A_), supA=ip.sup(A_),
                                                        infb=ip.inf(b_).reshape(-1, 1), supb=ip.sup(b_).reshape(-1, 1))
                print(tolmax)
                print(argmax)
                print(envs)

                draw_Tol(A_, b_, maxX, maxTol, f"Graphs/b-correction {i + 1}", fig.add_subplot(1, 3, i + 1, projection='3d'))
            fig.savefig(f"Graphs/b-corrections")

        case None:
            print("____Without correction____")
            fig = plt.figure(figsize=(18, 6))
            for i in range(len(As)):
                A_ = As[i]
                b_ = bs[i]

                emptiness_, maxX, maxTol = emptinessTol(A_, b_)
                print(emptiness_)
                
                print(ip.sup(b_).reshape(-1, 1))
                tolmax, argmax, envs, ccode = tolsolvty(infA=ip.inf(A_), supA=ip.sup(A_),
                                                        infb=ip.inf(b_).reshape(-1, 1), supb=ip.sup(b_).reshape(-1, 1))
                print(tolmax)
                print(argmax)
                print(envs)

                draw_Tol(A_, b_, maxX, maxTol, f"Graphs/No-correction {i + 1}", fig.add_subplot(1, 3, i + 1, projection='3d'))
            fig.savefig(f"Graphs/No-corrections")


run()
run("b")
run("A")

run("Ab")
plt.show()
# plt.show() # uncomment if you wanna watch all graphs
