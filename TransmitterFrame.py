import numpy as np 

import matplotlib.pyplot as plt


class TransmitterFrameField:

    def __init__(self, r , r0):
        m = np.array([1,0,0])
        # Calculate R and norm_R
        R = np.subtract(np.transpose(r), r0).T
        norm_R = np.sqrt(np.einsum("i...,i...", R, R)) 

        # Handle the case where the norm_R is zero to avoid division by zero
        norm_R[norm_R == 0] = np.inf

        # Calculate A (the magnetic field vector)
        m_dot_r = np.einsum("i,i...->...", m, R)
        self.B = (3 * R * m_dot_r / norm_R**5 - m[:, np.newaxis, np.newaxis, np.newaxis] / norm_R**3)  