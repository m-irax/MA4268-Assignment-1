# MA4268-Assignment-1
Discrete Cosine Transfrom (DCT) and Fast Fourier transform (FFT) are two essential transforms for many image processing tasks. 
The goal of this project is to implement two-dimensional DCT (FFT) and inverse DCT (FFT) for 2D images, either in Python or Matlab.

## Instructions Given
The two-dimensional DCT (FFT) can be done via first running 1D DCT (FFT) on each column (row), followed by running 1D DCT (FFT) on each row (column), which often is more eﬀicient than directly implementing 2D DCT (FFT). The following assumption is allowed in your implementation: the image size is of 2N × 2N where N is a positive integer.
You may NOT call or use any code from the following built-in functions in standard packages including numpy and scipy.
