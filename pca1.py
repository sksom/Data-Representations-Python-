import sys
import numpy as np

def main():

    #First arg is dataFile and second arg is labelsfile

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

    #Input ends and PCA algorithm starts
    R=np.dot((Xt.T),Xt)
    print("R=",R)

    #Now since R by it's property is symmetric but still to avoid any complication assume non-symmetric
    #Computing eigen values and eigen vectors of R
    evals, evecs = np.linalg.eig(R)
    print("evals=", evals, " evecs=", evecs)

    #The Eigen values and eigen vectors are not necessarily be sorted bcz eig() is used nd not eigh(), so sort them.
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    print("evals=", evals, " evecs=", evecs)  # evectors are the cols of evecs

    #now we need to extract the k=2 features out of the available features, here 4
    #Hence we need top 2 evectors(columns) corresponding to the top 2 max. evalues
    v1v2=evecs[:,[0,1]]
    print("v1v2=",v1v2)

    #Computing the projections
    proj=np.dot((v1v2.T),(Xt.T))
    #Need to transpose the proj because it's dimensions are 2xn
    proj=proj.T     # proj is now nx2 similar dimension as the input data
    print("Proj=",proj)

    # Write array to a file
    np.savetxt(first+'_pca1_output', proj, delimiter=',')


if __name__ == '__main__':
    main()