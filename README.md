# PDE simulation of bullet options
## _Supervised by Professor Lokmane Abbas Turki_

The goal of this project is to develop a deeper understanding of Partial Differential Equations (PDE) through the practical application of simulating bullet options. We aim to implement and critically analyze two different algorithms – the Thomas algorithm and Parallel Cyclic Reduction (PCR) – to solve tridiagonal systems. This simulation, based on the Crank-Nicolson scheme, will allow us to explore the nuances of these methods and their effectiveness in a financial context, specifically in the pricing of options.

![PDE Model](pde.png)

## Files and implemented functions
- src/pde.cu : first implementation of Thomas' algorithm and the PCR on 2 simple examples (one commented), this allows us to see that the algorithm does actually work before implementing it on more matrices
- src/pde_finalQ1.cu : second implementation on more randomely generated cases and also speed measurment
- src/pde_finalQ2.cu : second exercise
- src/pde_finalQ3.cu : third exercise
- data/timing_data_colab.txt : comparison values for Thomas' algorithm and PCR on the GPU of Google Colab
- data/timing_data_gpu.txt : comparison values for Thomas' algorithm and PCR on the GPU of the PPTI
- Sujets2023-2.pdf : full subject, our implementation focuses on part 3
- commandes.txt : execution commands for gnuplot

## Run the program

This program was made for a GPU environment. If you do not have access to a GPU, you can use https://colab.research.google.com/?hl=fr and before running the program select T4 GPU.

Here is how to run the program (on Google Colab) : 
```c
!nvcc -o <file_name> <file_name>.cu
!./file_name
```

## To run gnuplot for algorithm analysis

Install gnuplot

```sh
brew install gnuplot
```
```sh
gnuplot -p < commandes.txt
```

## Analysis

In our analysis we decided to look at randomely generated fixed size systems and varied the number of tridiagonal systems we were going to solve at the same time. As such, we varied the number of systems from 20 to 1024 (the maximum number of threads).
Here is a representation of our results :

![Comparison betweeen PCR and Thomas' method](graphs/Colabb.png)

We also executed on the GPU of the PPTI, we got : 

![Comparison betweeen PCR and Thomas' method](graphs/PPTII.png)

Which gives us better results because it has less trafic.

Here we can see that Thomas' algorithm is in O(n) and the PCR is in O(nlog(n)). 
The curve for Thomas' algorithm has a few irregularities which can be smoothed out by picking values of power 2. The method is illustrated here :

![Comparison betweeen PCR and Thomas' method (smooth)](graphs/PCR_Thomas2.png)

to get these results simply change the curve in pde_finalQ1.cu from 
```c 
for (int i = 20; i < 1024; i+=20)
``` 
to 
```c 
for (int i = 4; i <= 1024; i*=2)
```