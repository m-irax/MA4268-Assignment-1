#!/usr/bin/env python
# coding: utf-8


import math
import cmath
import numpy as np
from PIL import Image as image # to load test image
import matplotlib.pyplot as plt # to plot images


def lda(k):
    if k == 0:
        return math.sqrt(1/2)
    else:
        return 1


#DCT

def dct1d(im):
    # im is a sequence of numbers here; NOT A MATRIX
    N = len(im)
    C = np.zeros(N) # creating an empty array of numbers to store the dct values of the seq
    for n in range(N):
        for k in range(N):
            C[n] += im[k] * math.cos((k + 1/2) * math.pi * n/N)
        C[n] = C[n] * math.sqrt(2/N) * lda(n)
    return C

def dct2d(im): # im is a matrix here
    N = len(im)
    A = np.zeros([N,N])
    for i in range(N):
        A[:,i] = dct1d(im[:,i]) # do 1D DCT on the col of im first & plop into A

    B = np.zeros([N,N])
    for m in range(N):
        B[m,:] = dct1d(A[m,:]) # do 1D DCT on the row of A and plop into B
    return B



#iDCT

def idct1d(coef): # coef is a SEQUENCE, NOT MATRIX
    N = len(coef)
    f = np.zeros(N) # creating an empty array of numbers to store the idct values of the seq
    for k in range(N):
        for n in range(N):
            f[k] += coef[n] * math.cos((k + 1/2) * math.pi * n/N) * lda(n)
    f = f * math.sqrt(2/N)
    return f

def idct2d(coef): # coef is a matrix here
    N = len(coef)
    A = np.zeros([N,N])
    for i in range(N):
        A[:,i] = idct1d(coef[:,i]) # do 1D iDCT on the col of coef first & plop into A

    B = np.zeros([N,N])
    for m in range(N):
        B[m,:] = idct1d(A[m,:]) # do 1D iDCT on the row of A and plop into B
    return B



#FFT

def fft1d(im):
    # im is a sequence of numbers here
    N = len(im)
    if N == 1:
        return im
    else:
        f_even = im[0::2]
        f_odd = im[1::2]
        F_even = fft1d(f_even) # till left 1 element
        F_odd = fft1d(f_odd) # till left 1 element
        
        ######## end of recursion
        
        F = np.zeros(N, dtype = np.complex128)
        expon = cmath.exp((-1j * 2 * math.pi)/N)
        
        for k in range(N//2):
            F[k] = F_even[k] + F_odd[k] * expon**k
            F[k + N//2] = F_even[k] - F_odd[k] * expon**k
    return F

def fft2d(im):
    N = len(im)
    A = np.zeros([N,N], dtype = np.complex128)
    for i in range(N):
        A[i,:] = fft1d(im[i,:]) # do 1D FTT on the row of im first & plop into A

    B = np.zeros([N,N], dtype = np.complex128)
    for m in range(N):
        B[:,m] = fft1d(A[:,m]) # do 1D FFT on the column of A and plop into B
    return B



#iFFT

def ifft1d(coef):
    N = len(coef) # coef is a sequence here
    conj_F = fft1d(np.conj(coef)) # do 1D FFT on the conj of \hatf sequence
    f = 1/N * conj_F # conj of f
    return np.conj(f)

def ifft2d(coef):
    N = len(coef)
    A = np.zeros([N,N], dtype = np.complex128)
    for i in range(N):
        A[i,:] = ifft1d(coef[i,:]) #do 1D iFFT on the row of coef first & plop into A

    B = np.zeros([N,N])
    for m in range(N):
        B[:,m] = ifft1d(A[:,m]) #do 1D iFFT on the column of A and plop into B
    return B