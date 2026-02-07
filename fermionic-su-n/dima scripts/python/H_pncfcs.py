import numpy as np
from overlap import overlap

def H_pncfcs(z, I1, I2, debug = False):
    N = z.shape[1]
    S = np.zeros((N,N), dtype=complex)
    E1 = np.zeros((N,N), dtype=complex)
    E2 = np.zeros((N,N), dtype=complex)

    if debug:
        E_upup = np.zeros((N,N), dtype=complex)
        E_downdown = np.zeros((N,N), dtype=complex)
        E_mixed = np.zeros((N,N), dtype=complex)

    for n in range(N):
        for m in range(N):
            z1 = z[:,n]
            z2 = z[:,m]

            z1u = z1[:5]
            z1d = z1[5:]
            z2u = z2[:5]
            z2d = z2[5:]

            Su = overlap(z1u*z2u, 3)
            Sd = overlap(z1d*z2d, 3)
            S[n,m] = Su*Sd

            F1 = np.zeros((10,10), dtype=complex)
            F2 = np.zeros((10,10,10,10), dtype=complex)

            # one-electron (up)
            for i in range(5):
                for j in range(5):
                    z1u2 = z1u.copy()
                    z2u2 = z2u.copy()
                    val = z1u2[i]*z2u2[j]
                    z1u2[i] = 0
                    z2u2[j] = 0
                    z1u2[:i+1] *= -1
                    z2u2[:j+1] *= -1
                    F1[i,j] = val * overlap(z1u2*z2u2,2) * Sd

            # one-electron (down)
            for i in range(5):
                for j in range(5):
                    z1d2 = z1d.copy()
                    z2d2 = z2d.copy()
                    val = z1d2[i]*z2d2[j]
                    z1d2[i] = 0
                    z2d2[j] = 0
                    z1d2[:i+1] *= -1
                    z2d2[:j+1] *= -1
                    F1[i+5,j+5] = val * overlap(z1d2*z2d2,2) * Su

            E1[n,m] = np.sum(F1 * I1)

            # two-electron (up-up)
            """for i1 in range(5):
                for i2 in range(i1+1,5):
                    for j1 in range(5):
                        for j2 in range(j1+1,5):
                            z1u2 = z1u.copy()
                            z2u2 = z2u.copy()
                            val = z1u2[i1]*z1u2[i2]*z2u2[j1]*z2u2[j2]
                            z1u2[i1]=0; z1u2[i2]=0
                            z2u2[j1]=0; z2u2[j2]=0
                            z1u2[i1:i2+1] *= -1
                            z2u2[j1:j2+1] *= -1
                            base = val * overlap(z1u2*z2u2,1) * Sd
                            F2[i1,i2,j2,j1] = base
                            F2[i2,i1,j2,j1] = -base
                            F2[i1,i2,j1,j2] = -base
                            F2[i2,i1,j1,j2] = base

            # down-down is identical structure
            # up-down block is also identical (see your Matlab)"""

            for i1 in range(5):
                for i2 in range(i1 + 1, 5):
                    for j1 in range(5):
                        for j2 in range(j1 + 1, 5):

                            z1u2 = z1u.copy()
                            z2u2 = z2u.copy()

                            F2[i1,i2,j2,j1] = z1u2[i1] * z1u2[i2] * z2u2[j1] * z2u2[j2]

                            z1u2[i1]=0
                            z1u2[i2]=0
                            z2u2[j1]=0
                            z2u2[j2]=0

                            z1u2[i1:i2]*=-1
                            z2u2[j1:j2]*=-1

                            F2[i1,i2,j2,j1] = F2[i1,i2,j2,j1] * overlap(z1u2*z2u2,1) * Sd
                            F2[i2,i1,j2,j1] = -F2[i1,i2,j2,j1]
                            F2[i1,i2,j1,j2] = -F2[i1,i2,j2,j1]
                            F2[i2,i1,j1,j2] = F2[i1,i2,j2,j1]

            for i1 in range(5):
                for i2 in range(i1 + 1, 5):
                    for j1 in range(5):
                        for j2 in range(j1 + 1, 5):

                            z1d2 = z1d.copy()
                            z2d2 = z2d.copy()

                            F2[i1+5,i2+5,j2+5,j1+5] = z1d2[i1] * z1d2[i2] * z2d2[j1] * z2d2[j2]

                            z1d2[i1]=0
                            z1d2[i2]=0
                            z2d2[j1]=0
                            z2d2[j2]=0

                            z1d2[i1:i2]*=-1
                            z2d2[j1:j2]*=-1

                            F2[i1+5,i2+5,j2+5,j1+5] = F2[i1+5,i2+5,j2+5,j1+5] * overlap(z1d2*z2d2,1) * Su
                            F2[i2+5,i1+5,j2+5,j1+5] = -F2[i1+5,i2+5,j2+5,j1+5]
                            F2[i1+5,i2+5,j1+5,j2+5] = -F2[i1+5,i2+5,j2+5,j1+5]
                            F2[i2+5,i1+5,j1+5,j2+5] = F2[i1+5,i2+5,j2+5,j1+5]

            for i1 in range(5):
                for i2 in range(5, 10):
                    for j1 in range(5):
                        for j2 in range(5, 10):

                            z1u2 = z1u.copy()
                            z1d2 = z1d.copy()
                            z2u2 = z2u.copy()
                            z2d2 = z2d.copy()

                            F2[i1,i2,j2,j1] = z1u2[i1] * z1d2[i2-5] * z2u2[j1] * z2d2[j2-5]

                            z1u2[i1]=0
                            z1d2[i2-5]=0
                            z2u2[j1]=0
                            z2d2[j2-5]=0

                            z1u2[0:i1]=-z1u2[0:i1]
                            z1d2[0:i2-5]=-z1d2[0:i2-5]
                            z2u2[0:j1]=-z2u2[0:j1]
                            z2d2[0:j2-5]=-z2d2[0:j2-5]

                            F2[i1,i2,j2,j1] = F2[i1,i2,j2,j1] * overlap(z1u2*z2u2,2) * overlap(z1d2*z2d2,2)
                            F2[i2,i1,j2,j1] = -F2[i1,i2,j2,j1]
                            F2[i1,i2,j1,j2] = -F2[i1,i2,j2,j1]
                            F2[i2,i1,j1,j2] = F2[i1,i2,j2,j1]

            if debug:
                E_upup[n,m] = np.sum(F2[:5,:5,:5,:5] * I2[:5,:5,:5,:5])
                E_downdown[n,m] = np.sum(F2[5:,5:,5:,5:] * I2[5:,5:,5:,5:])
                E_mixed[n,m] = (np.sum(F2[:5,5:,:5,5:] * I2[:5,5:,:5,5:]) + np.sum(F2[5:,:5,5:,:5] * I2[5:,:5,5:,:5]) +
                            np.sum(F2[:5,5:,5:,:5] * I2[:5,5:,5:,:5]) + np.sum(F2[5:,:5,:5,5:] * I2[5:,:5,:5,5:]))

                print("partitioned sum:",(
                    np.sum(F2[:5,:5,:5,:5] * I2[:5,:5,:5,:5]) +
                    np.sum(F2[5:,5:,5:,5:] * I2[5:,5:,5:,5:]) +
                    np.sum(F2[:5,5:,:5,5:] * I2[:5,5:,:5,5:]) +
                    np.sum(F2[5:,:5,5:,:5] * I2[5:,:5,5:,:5]) +
                    np.sum(F2[:5,5:,5:,:5] * I2[:5,5:,5:,:5]) +
                    np.sum(F2[5:,:5,:5,5:] * I2[5:,:5,:5,5:]) )
                    )
                print("Whole sum:", np.sum(F2 * I2))


            E2[n,m] = np.sum(F2 * I2)
            """for i in range(5):
                for j in range(i+1,5):
                    for k in range(5):
                        for l in range(k+1,5):
                            E2[n,m] += F2[i,j,l,k] * I2[i,j,k,l]"""

    if debug:
        print("E1 =", E1)
        print("E2 =", E2/2)
        print("E_nuc =", 1.8*S)

        print("Up-up:", E_upup)
        print("Down-down:", E_downdown)
        print("Mixed-spin:", E_mixed)

    H = E1 + E2/2 + 1.8*S
    return H, S

