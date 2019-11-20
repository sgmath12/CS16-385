import numpy as np
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pdb

# write your implementation here
data = np.load('../data/pnp.npz',allow_pickle=True)

for k in data.keys():
    print (k)


x = data['x']
X = data['X']
N = x.shape[0]
im = data['image']
cad = data['cad']


P = sub.estimate_pose(x,X)
K,R,t = sub.estimate_params(P)

hX = np.hstack([X,np.ones([N,1])])
projectPt = P@hX.T

projectPt = np.round((projectPt/projectPt[-1]).T).astype('int')
projectPt = projectPt[:,:-1]

implot = plt.imshow(im)

# put a blue dot at (10, 20)
plt.scatter([10], [20])
# put a red dot, size 40, at 2 locations:
plt.scatter(x=projectPt[:,0], y=projectPt[:,1], c='r', s=40)
plt.show()

plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

vert = cad['vertices'][0][0]
faces = cad['faces'][0][0]

x = cad['vertices'][0][0][:,0]
y = cad['vertices'][0][0][:,1]
z = cad['vertices'][0][0][:,2]



cadVertices = np.concatenate([x,y,z]).reshape([3,-1]).T
Rotated = R@cadVertices.T
Rotated = Rotated.T


#ax.scatter(x, y, z, c='black', s = 1, marker='*')
ax.add_collection3d(Poly3DCollection(vert[faces-1],alpha = 0.2,color = 'black'))
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.add_collection3d(Poly3DCollection(Rotated[faces-1],alpha = 0.2,color = 'black'))
plt.show()
plt.close()


N = cadVertices.shape[0]
hcadVertices = np.hstack([cadVertices,np.ones([N,1])])
projectPt = P@hcadVertices.T
projectPt = np.round((projectPt/projectPt[-1]).T).astype('int')
projectPt = projectPt[:,:-1]

implot = plt.imshow(im)

# put a blue dot at (10, 20)
plt.scatter([10], [20])
# put a red dot, size 40, at 2 locations:
plt.scatter(x=projectPt[:,0], y=projectPt[:,1],c = 'b' ,marker='o', s=5)
plt.show()

