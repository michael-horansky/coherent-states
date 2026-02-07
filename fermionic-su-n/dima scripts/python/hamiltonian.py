import numpy as np

def hamiltonian(z, I1, I2):
    N = z.shape[2]
    H = np.zeros((N,N))
    S = np.zeros((N,N))

    for n in range(N):
        for m in range(N):
            D = np.zeros((10,10))
            for i in range(6):
                D[i,i] = 1
            for i in range(6,10):
                D[i,i] = -1

            for i in range(3):
                for j in range(2):
                    D[i,j+6]   = z[j,i,n]
                    D[i+3,j+8] = z[j,i,n]
                    D[j+6,i]   = z[j,i,m]
                    D[j+8,i+3] = z[j,i,m]

            detD = np.linalg.det(D)
            S[n,m] = detD
            fm = np.linalg.inv(D) * detD

            Fm = np.zeros((10,10,10,10))
            for i1 in range(9):
                for i2 in range(i1+1,10):
                    for j1 in range(9):
                        for j2 in range(j1+1,10):
                            Dm = D.copy()
                            Dm[i1,:]=0; Dm[i2,:]=0
                            Dm[:,j1]=0; Dm[:,j2]=0
                            Dm[i1,j1]=1; Dm[i2,j2]=1
                            Fmc = np.linalg.det(Dm)

                            if i2==j2 and i2>5:
                                Fmc += fm[j1,i1]
                                if i1==j1 and i1>5:
                                    Fmc += detD
                            if i2==j1 and i2>5:
                                Fmc -= fm[j2,i1]
                            if i1==j1 and i1>5:
                                Fmc += fm[j2,i2]
                            if i1==j2 and i1>5:
                                Fmc -= fm[j1,i2]

                            Fm[i1,i2,j1,j2] = Fmc
                            Fm[i2,i1,j2,j1] = Fmc
                            Fm[i1,i2,j2,j1] = -Fmc
                            Fm[i2,i1,j1,j2] = -Fmc

            inx = np.array([0,1,2,5,6,7,3,4,8,9])
            Fm2 = np.zeros_like(Fm)
            for a in range(10):
                for b in range(10):
                    for c in range(10):
                        for d in range(10):
                            Fm2[inx[a],inx[b],inx[c],inx[d]] = -Fm[a,b,c,d]

            fm1 = np.zeros((10,10))
            for i in range(6,10):
                fm[i,i] += detD
            for a in range(10):
                for b in range(10):
                    fm1[inx[a],inx[b]] = fm[a,b]

            E1 = np.sum(fm1 * I1)
            E2 = np.sum(Fm2 * I2)
            H[n,m] = E1 + E2/2 + 1.8*S[n,m]

    return H, S

