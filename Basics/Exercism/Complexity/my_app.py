import argparse
import numpy as np
import matplotlib.pyplot as plt
# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--map_params",
  nargs="*",
  type=float,  # any type/callable can be used here
  default=[],
)

# parse the command line
args = CLI.parse_args()

x0 = args.map_params[0]
rmin = args.map_params[1]
rmax = args.map_params[2]
rstep = args.map_params[3]
n = int(args.map_params[4])
k = int(args.map_params[5])

def logistic(r, x):
    return r * x * (1 - x)

rIter = int((rmax-rmin)/rstep)
r = np.linspace(rmin, rmax, rIter)
fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
x = x0*np.ones(rIter)
for i in range(n):
    x = logistic(r, x)
    #display bifurcation diagram for iterates after n-k
    if i >= k:     
        plt.plot(r, x, ',k', alpha=0.25)
    
plt.xlim(rmin, rmax)
plt.ylim(0,1)
plt.title("Bifurcation diagram")
plt.xlabel("r", fontsize="18")
plt.ylabel("x", fontsize="16")
plt.xticks(np.arange(rmin,rmax+0.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.tight_layout()
plt.show()

'''
Version 1
def logistic(r, x0, n):
    xn = [x0]
    for i in range(n):
        xn.append(r*xn[i]*(1-xn[i]))
    return np.round(xn,decimals=4)

rIter = int((rmax-rmin)/rstep)
r = np.linspace(rmin, rmax, rIter)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
for j in range(rIter):
    x = logistic(r[j], x0, n)
    # display bifurcation diagram for iterates after k
    plt.plot(r[j]*np.ones(k), x[(n-k):-1], ',k', alpha=0.25)
    
plt.xlim(rmin, rmax)
plt.title("Bifurcation diagram")
plt.tight_layout()
plt.show()
'''

'''
This is done with the python cookbook and seems more scalable
def logistic(r, x):
    return r * x * (1 - x)

n = 10000
r = np.linspace(2.5, 4.0, n)

iterations = 1000
last = 100

x = 1e-5 * np.ones(n)

lyapunov = np.zeros(n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               sharex=True)
for i in range(iterations):
    x = logistic(r, x)
    # We compute the partial sum of the
    # Lyapunov exponent.
    lyapunov += np.log(abs(r - 2 * r * x))
    # We display the bifurcation diagram.
    if i==1:
            print(x)
    if i >= (iterations - last):
        ax1.plot(r, x, ',k', alpha=.25)

ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.5)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / iterations,
         '.k', alpha=.5, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / iterations,
         '.r', alpha=.5, ms=.5)
ax2.set_xlim(2.5, 4)
ax2.set_ylim(-2, 1)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()
'''