import numpy as np

def _pos2index(trj, num_grid, x_min, dx):
    x_grid = np.around((trj - x_min)/dx)
    
    trj_idx = np.zeros((trj.shape[0],), dtype=np.int32)

    num_beads = trj.shape[1]
    for i in range(num_beads):
        trj_idx += np.int32(x_grid[:, i] * num_grid**i)
    
    return trj_idx



def DFR_plug_in(trjs):
    num_trjs, trj_len = trjs.shape

    Transition = np.zeros((6,6), dtype=np.int32)

    for i in range(num_trjs):
        trj = trjs[i]

        for j in range(trj_len-1):
            x_1, x_2 = trj[j], trj[j+1]

            Transition[x_1, x_2] += 1

    KLD = 0
    
    KLD = 0
    Transition += 1
    Transition_rev = np.transpose(Transition)
    
    KLD += np.sum(Transition * np.log(Transition/Transition_rev))
    # for i in range(6):
    #     for j in range(6):
    #         if Transition[j, i] != 0 and Transition[i, j] != 0:
    #             KLD += Transition[i, j] * np.log(Transition[i, j]/Transition[j, i])

    return KLD/(np.sum(Transition))


def multibeads_plug_in(trjs, dt, num_grid=20, test_trj=None):
    num_trjs, trj_len, num_beads = trjs.shape

    grid = 1
    for i in range(num_beads): grid *= num_grid

    Transition = np.zeros((grid, grid), dtype=np.int32)

    x_max = np.max(np.max(trjs, axis=1), axis=0)
    x_min = np.min(np.min(trjs, axis=1), axis=0)

    dx = (x_max - x_min)/(num_grid-1)

    for i in range(num_trjs):
        trj = trjs[i]

        trj_idx = _pos2index(trj, num_grid, x_min, dx)

        for j in range(trj_len-1):
            Transition[trj_idx[j], trj_idx[j+1]] += 1

    KLD = 0

    num_trans = 0
    for i in range(grid):
        for j in range(grid):
            if Transition[j, i] != 0 and Transition[i, j] != 0:
                KLD += Transition[i, j] * np.log(Transition[i, j]/Transition[j, i])
                num_trans += Transition[i, j]

    print(num_trans/np.sum(Transition))

    if test_trj is None:
        return KLD/(num_trans * dt)
    
    else:
        etpy = np.zeros((trj_len,))
        for j in range(trj_len-1):
            if Transition[trj_idx[j], trj_idx[j+1]] != 0 and Transition[trj_idx[j+1], trj_idx[j]] != 0:
                etpy[j] = np.log(Transition[trj_idx[j], trj_idx[j+1]]/Transition[trj_idx[j+1], trj_idx[j]])

        return KLD/(num_trans * dt), etpy


