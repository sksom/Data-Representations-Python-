import sys
import numpy as np

def main():

    # First arg is dataFile and second arg is labelsfile

    if len(sys.argv) != 3:
        print(sys.argv[0], "takes 2 arguments. Not ", len(sys.argv) - 1)
        sys.exit()

    first = sys.argv[1]
    second = sys.argv[2]
    print(sys.argv[0], "args are:", first, second)

    # Read numpy array from a file

    # The file f1.txt has two rows. First row is 1  2 seond row is 3 4
    Y = np.genfromtxt(second)  # default delimiter is space
    print("Y=", Y)

    # The file f2.txt : First row is 1, 2 seond row is 3, 4
    Xt = np.genfromtxt(first, delimiter=',', autostrip=True)  # trip spaces
    print("X=", Xt)

    #RESHAPING
    n, m = Xt.shape
    # Y is a vector, so reshape it to matrix
    Y = Y.reshape(n, 1)
    # Get the index where the value changes in Y
    idx = np.where(np.diff(Y[:, 0]))[0] + 1
    print("Change Idx:", idx)

    #The Calculation of Within Class Scatter matrix starts from here
    # No. of points in each class
    print("Total no. of points=",n)
    m1=idx[0]
    m2=idx[1]-idx[0]
    m3=n-idx[1]
    print("m1=", m1)
    print("m2=", m2)
    print("m3=", m3)
    #mu's for each class
    mu=np.mean(Xt, axis=0)
    mu=mu.reshape(m,1)
    print("Global mu:",mu)
    mu1 = np.mean(Xt[0:idx[0],:], axis=0)
    mu1 = mu1.reshape(m, 1)
    print("mu1:", mu1)
    mu2 = np.mean(Xt[m1:idx[1], :], axis=0)
    mu2 = mu2.reshape(m, 1)
    print("mu2:", mu2)
    mu3 = np.mean(Xt[idx[1]:n, :], axis=0)
    mu3 = mu3.reshape(m, 1)
    print("mu3:", mu3)

    #Calculating the Scatter matrix for each group to finally constitute the W
    s1=np.dot((Xt[0:idx[0],:].T-mu1),(Xt[0:idx[0],:].T-mu1).T)
    s2 = np.dot((Xt[idx[0]:idx[1], :].T - mu2), (Xt[idx[0]:idx[1], :].T - mu2).T)
    s3 = np.dot((Xt[idx[1]:n, :].T - mu3), (Xt[idx[1]:n, :].T - mu3).T)
    W=s1+s2+s3
    print("W=",W)

    #Calculating B
    M = np.dot((Xt.T) - mu, ((Xt.T) - mu).T)
    print("M=", M)

    B=M-W

    ratio=B/W

    # Now since W by it's property is symmetric but still to avoid any complication assume non-symmetric
    # Computing eigen values and eigen vectors of W
    evals, evecs = np.linalg.eig(ratio)
    print("evals=", evals, " evecs=", evecs)

    # The Eigen values and eigen vectors are not necessarily be sorted bcz eig() is used nd not eigh(), so sort them.
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    print("evals=", evals, " evecs=", evecs)  # evectors are the cols of evecs

    #Since we want to minimize the W we need the smallest eval
    # now we need to extract the 1 last evec out of the available evecs
    # Hence we need last 1 evector(column) corresponding to the last 1 min. evalues
    v = evecs[:, [0,1]]
    print("So the direction that minimizes the within class scatter is =", v)
    # Projection
    proj = np.dot(v.T, Xt.T)
    proj = proj.T
    print("So the projection that minimizes the within class scatter is =", proj)

    np.savetxt(first + '_scatter3_output', proj, delimiter=',')

if __name__ == '__main__':
    main()