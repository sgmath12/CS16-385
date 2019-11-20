import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load('../data/some_corresp.npz')
im1,im2 = io.imread('../data/im1.png'),io.imread('../data/im2.png')
M = max(im1.shape[0],im1.shape[1])
height = im1.shape[0]
width = im1.shape[1]
# 2. Run eight_point to compute F
""" 
data['pts1'][:,0] =  width - data['pts1'][:,0]
data['pts2'][:,0] =  width - data['pts2'][:,0]
 """
F = sub.eight_point(data['pts1'],data['pts2'],M)
#F = hlp.refineF(F,data['pts1']/M,data['pts2']/M)

T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
F = np.matmul(F,T)
#T = np.array([[M,0,0],[0,M,0],[0,0,1]])
F = np.matmul(T.T,F)
F = hlp.refineF(F,data['pts1'],data['pts2'])

#F = np.array([[0,0,0], [0,0,-0.0011], [0,0.0011,0.0045]])

#hlp.displayEpipolarF(im1,im2,F)

# 3. Load points in image 1 from data/temple_coords.npz
data = np.load('../data/temple_coords.npz')
pts1 = data['pts1']

# pts1[:0] = width - pts1[:0]

# 4. Run epipolar_correspondences to get points in image 2

#hlp.epipolarMatchGUI(im1,im2,F)

# 5. Compute the camera projection matrix P1
pts2 = sub.epipolar_correspondences(im1,im2,F,pts1)

import matplotlib.pyplot as plt
im = plt.imread('../data/im2.png')
implot = plt.imshow(im)

# put a blue dot at (10, 20)
plt.scatter([10], [20])

# put a red dot, size 40, at 2 locations:
plt.scatter(x=pts2[:,0], y=pts2[:,1], c='b', s=40)

plt.show()

Intrinsic = np.load('../data/intrinsics.npz')

P1 = np.hstack([Intrinsic['K1'],np.zeros([3,1])])
# 6. Use camera2 to get 4 camera projection matrices P2
E = sub.essential_matrix(F,Intrinsic['K1'],Intrinsic['K2'])

P2s = hlp.camera2(E)

#pdb.set_trace()

# 7. Run triangulate using the projection matrices
bestWorldPts,bestNumInfront = None,0
bestI = None
for i in range(4):
    worldPts,numInfrontBoth = sub.triangulate(P1,pts1,Intrinsic['K2']@P2s[:,:,i],pts2)
    if bestNumInfront < numInfrontBoth :
        bestWorldPts = worldPts
        bestNumInfront = numInfrontBoth 
        bestI = i

# 8. Figure out the correct P2
# pdb.set_trace()

data = np.load('../data/some_corresp.npz')
pts1,pts2= data['pts1'], data['pts2']
""" 
pts1[:,0] = width - pts1[:,0]
pts2[:,0] = width - pts2[:,0] 
"""
P2 = Intrinsic['K2']@P2s[:,:,2]
pts3d,_ = sub.triangulate(P1,pts1,P2,pts2)

pts1_proj = P1 @ pts3d.T
pts2_proj = P2 @ pts3d.T
pts1_proj = pts1_proj/pts1_proj[-1] 
pts2_proj = pts2_proj/pts2_proj[-1]

pts1_proj = np.round((pts1_proj.T)[:,:-1])
pts2_proj = np.round((pts2_proj.T)[:,:-1])

err1 = np.sqrt(np.sum(np.square(pts1 - pts1_proj),axis = 1)).mean()
err2 = np.sqrt(np.sum(np.square(pts2 - pts2_proj),axis = 1)).mean()
total_err = err1 + err2
print (err1,err2)
# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = bestWorldPts[:,0]
y = bestWorldPts[:,1]
z = bestWorldPts[:,2]


ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

k = bestWorldPts.mean(axis=0)
ax.set_xlim3d(k[0]-1,k[0]+1)
ax.set_ylim3d(k[1]-1,k[1]+1)
ax.set_zlim3d(k[2]-1,k[2]+1)
plt.show()
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = np.eye(3)
t1 = np.zeros([3,1])
R2 = P2s[:,:,bestI][:,:3]
t2 = P2s[:,:,bestI][:,3:4]
np.savez('../data/extrinsics.npz', R1=R1, t1=t1,R2 = R2, t2 = t2)
