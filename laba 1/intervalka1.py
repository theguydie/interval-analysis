import numpy as np

def Print(A):
    for str in A:
        print(str)
    return 0

def IntMult(a, b):
    res_min = a[0]*b[0]
    res_max = a[0]*b[0]

    # remember indeces of elems in result multiply (MIN = 0/MAX = 1)
    indeces = [[0, 0], [0, 0]]

    for i in range(2):
        for j in range(2):
            if a[i]*b[j] < res_min:
                res_min = a[i]*b[j]
                indeces[0] = [i, j]
            if a[i]*b[j] > res_max:
                res_max = a[i]*b[j]
                indeces[1] = [i, j]

    return [res_min, res_max], indeces

def main():
    # TEST 1: suggested in task
    midA = [[1.05, 0.95],
         [1, 1]]
    radA = [[1, 1],
            [1, 1]]

    # # TEST 2: self-created
    # midA = [[2.5, 0.6],
    #      [5.7, 2.8]]
    # radA = [[1, 1],
    #         [1, 1]]
    
    # accuracy
    eps = 0.0001

    # A - matrix of intervals
    A = [[[midA[0][0], midA[0][0]],[midA[0][1], midA[0][1]]],
         [[midA[1][0], midA[1][0]],[midA[1][1], midA[1][1]]]]

    # indeces are for recover the matrix with det = 0
    indeces1 = [[0, 0], [0, 0]]
    indeces2 = [[0, 0], [0, 0]]

    detA = [midA[0][0]*midA[1][1] - midA[1][0]*midA[0][1], midA[0][0]*midA[1][1] - midA[1][0]*midA[0][1]]

    # exit-condition: one border are close to zero ( |detA| < eps )
    # calculate det's interval here (MDP)
    stepp = max(midA[0][0], midA[0][1], midA[1][0], midA[1][1])
    alpha = 0
    while not (abs(detA[0]) < eps or abs(detA[1]) < eps):
        for i in range(len(A)):
            for j in range(len(A[i])):
                A[i][j][0] = midA[i][j] - alpha*radA[i][j]
                A[i][j][1] = midA[i][j] + alpha*radA[i][j]

        intmult1, indeces1 = IntMult(A[0][0], A[1][1])
        intmult2, indeces2 = IntMult(A[0][1], A[1][0])
        detA = [intmult1[0] - intmult2[1], intmult1[1] - intmult2[0]]
        if (detA[0]*detA[1] < 0):
            alpha -= stepp
        else:
            alpha += stepp
        stepp = stepp/2

    

    # FIND OUT A'
    A_acc = [[0, 0],
             [0, 0]]

    # detA = [min(a00*a11) - max(a01*a10), max(a00*a11) - min(a01*a10)]
    # a00...a11 can be recovered if we know which border reach zero
 
    # zero-reach-left-border case: min a00*a11 & max a01*a10 
    if abs(detA[0]) < eps:
        A_acc[0][0] = A[0][0][indeces1[0][0]]
        A_acc[1][1] = A[1][1][indeces1[0][1]]

        A_acc[0][1] = A[0][1][indeces2[1][0]]
        A_acc[1][0] = A[1][0][indeces2[1][1]]

    # zero-reach-right-border case: max a00*a11 & min a01*a10 
    if abs(detA[1]) < eps:
        A_acc[0][0] = A[0][0][indeces1[1][0]]
        A_acc[1][1] = A[1][1][indeces1[1][1]]

        A_acc[0][1] = A[0][1][indeces2[0][0]]
        A_acc[1][0] = A[1][0][indeces2[0][1]]

    print("alpha is: ", alpha)
    print()
    print("detA is: ", detA)
    print()
    print("A' is: ")
    Print(A_acc)

    return 0

if __name__ == "__main__":
    main()