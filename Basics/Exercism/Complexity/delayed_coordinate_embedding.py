import argparse
import numpy as np
import matplotlib.pyplot as plt

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--map_params",
  nargs="*",
  type=int,  # any type/callable can be used here
  default=[],
)

# parse the command line
args = CLI.parse_args()

tau = args.map_params[0]
m = args.map_params[1]

amplitudeArray = np.loadtxt('amplitude2.dat')

n = int(len(amplitudeArray)/m)
x = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        x[i, j] = amplitudeArray[i+j*tau]
    
#print(x)


fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(x[:,0],x[:,1])
plt.title("Reconstruction State Space")
plt.xlabel("x_0", fontsize="18")
plt.ylabel("x_2", fontsize="16")
plt.tight_layout()
plt.show()
    
eps = 0.5
[minx, minz] = np.amin(x, axis=0)
[maxx, maxz] = np.amax(x, axis=0)
n = len(x[:,0])
epsMat = np.zeros( (int(np.ceil((maxx-minx)/eps))+1, int(np.ceil((maxz-minz)/eps))+1)  )
#create a map that takes the x-coordinate to an index in the matrix
#create a map that takes the z-coordinate to an index in the matrix
for i in range(n):
# 1. xcurr -> ceiling[(xcurr-minx)/eps]
     xcurr = int(np.ceil((x[i,0]-minx)/eps))
# 2. repeat for z-coordinate
     zcurr = int(np.ceil((x[i,1]-minz)/eps))
# 3. store value in matrix and set equal to 1
     epsMat[xcurr, zcurr] = 1
# 4. repeat for all coordinates
# 5. take the frobenius norm of the matrix
Neps = np.sum(epsMat)
print(Neps)