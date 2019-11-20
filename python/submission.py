"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import scipy.optimize
import scipy.signal
import pdb


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""



def eight_point(pts1, pts2, M):
     # 0. normalize 
    pts1,pts2 = pts1 / M, pts2 / M
    # 1. construct A marix : N,9
    ones = np.ones([pts1.shape[0],1])
    Apts1 , Apts2 = np.hstack((pts1,ones)),np.hstack((pts2,ones))
    Apts1 , Apts2 = np.repeat(Apts1,3,axis =1),np.hstack((Apts2,Apts2,Apts2))
    A =  Apts1 * Apts2
  
    #print (A[0])
    # 2. find SVD of A and get F
    U,S,V = np.linalg.svd(A) 
    F = V[-1]
    F = F.reshape([3,3])
    
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U@np.diag(S)@V
    #F = U.dot(np.diag(S).dot(V))
    
    # 3. force the matrix has rank 2
    
    return F

    # 6. visualize the correctness of estimated Fundamental matrix
   


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def _similarity(im1,im2,pt1,compares,w):
    
    y,x = pt1[0],pt1[1]

    npad = ((w//2, w//2), (w//2, w//2), (0, 0))
    im1pad = np.pad(im1,npad,mode = 'constant')
    x_,y_ = x + w//2, y + w//2

    patch1 = im1pad[x_ - w//2:x_+w//2 + 1,y_ - w//2:y_+ w//2 + 1].reshape(-1)
    A = ((patch1 - compares)**2).sum(axis = 1)
    
    return (A.argmin())

def similarity(im1,im2,pts1,pts2):

    d = 0
    windowSize = 5
    windows = np.arange(windowSize) - windowSize//2 
    pts1 = pts1.reshape([-1])
    coords = [(x,y) for x in windows for y in windows]

    for wx,wy in coords:
        if 0 <= pts1[0] + wx < im1.shape[0] and 0 <= pts1[1] + wy <im1.shape[1] \
            and 0 <= pts2[0] + wx < im2.shape[0] and 0 <= pts2[1] + wy <im2.shape[1] : 
            d += ((im1[pts1[0] + wx,pts1[1]+wy] - im2[pts2[0]+wx,pts2[1] + wy])**2).sum()

    return d
        

def epipolar_correspondences(im1, im2, F, pts1):

    # homogeneous 

    pts2 = np.array([0,0])
    w = 19
    npad = ((w//2, w//2), (w//2, w//2), (0, 0))
    im2pad = np.pad(im2,npad,mode = 'constant')
    pts2 = np.array([0,0])
    
    # get candidate
    for pt1 in pts1:
        hpt1 = np.append(pt1,1).reshape([-1,1])
        line = np.matmul(F,hpt1)
       
        candidates = np.zeros([1,2])
        compares = np.zeros([w*w*4])

        m = -line[0]/line[1]
        b = -line[2]/line[1]
        x = np.arange(0,im2.shape[1])
        y = np.round(m*x+b)
        
        x = x[y >= 0]
        y = y[y >= 0]
        x = x[y < im2.shape[0]]
        y = y[y < im2.shape[0]].astype('int')

        for px,py in zip(x,y):
            # im2 padding
            x_,y_ = px + w//2, py + w//2
            compares = np.vstack([compares,im2pad[y_ - w//2:y_+w//2 + 1,x_ - w//2:x_+ w//2 + 1].reshape(-1)])
            candidates = np.vstack([candidates,[px,py]])
           
        candidates = candidates[1:]
        compares = compares[1:]
        idx = _similarity(im1,im2,pt1,compares,w)
        pts2 = np.vstack([pts2,candidates[idx]])
 
     
    return pts2[1:]

'''
def epipolar_correspondences(im1, im2, F, pts1):
    
    
    # homogeneous 
    
    bestCorrespondence = None
    pts2 = np.array([0,0])
    # get candidate
    for pt1 in pts1:
        bestD = float('Inf')
        hpt1 = np.append(pt1,1).reshape([-1,1])
        line = np.matmul(F,hpt1)
        candidates = np.array([[0,0]])
        for x in range(im2.shape[0]):
            y = np.clip(-(line[0]*x + line[2])/line[1],0,im2.shape[1]-1).astype('int')[0]
            candidates = np.vstack([candidates,[x,y]])
            d = similarity(im1,im2,pt1,[x,y])
            if bestD > d:
                bestD = d
                bestCorrespondence = np.array([x,y])
        
        
        pts2 = np.vstack([pts2,bestCorrespondence])
   
    return pts2[1:]
'''

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    return K2.T @ F @ K1

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def countInfront(P1,P2,worldPts):
    '''
    wordsPts = (N,4)
    '''
    imgPts1 , imgPts2 = P1@(worldPts.T), P2 @(worldPts.T)
    numInfrontBoth = ((imgPts1[-1] >= 0)*(imgPts2[-1] >= 0)).sum()
    return numInfrontBoth

def triangulate(P1, pts1, P2, pts2):
    # 
    N = pts1.shape[0]
    worldPts = np.zeros([N,4])
    for i, (pt1 ,pt2) in enumerate(zip(pts1,pts2)):
        x,y = pt1[0], pt1[1]
        x_,y_ = pt2[0], pt2[1]

        #print (pt1,pt2)
        #x,y = pt1[0],pt1[1]
        #x_,y_ = pt2[0],pt2[1]

        A = y*P1[2] - P1[1]
        A = np.vstack([A,P1[0] - x*P1[2]])
        A = np.vstack([A,y_*P2[2] - P2[1]])
        A = np.vstack([A,P2[0] - x_*P2[2]])

        #print (A)
        #pdb.set_trace()

        _,S,V = np.linalg.svd(A.T @ A)  
        worldPt = V[np.argmin(np.abs(S))]
        
        #worldPt = V[-1]
        worldPt = worldPt/worldPt[-1]
        worldPts[i] = worldPt
        #pdb.set_trace()


    return worldPts,countInfront(P1,P2,worldPts)
    

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # 1. Compute the optical centers c1, c2
    c1 = -np.linalg.inv((np.matmul(K1,R1)))@np.matmul(K1,t1)
    c2 = -np.linalg.inv((np.matmul(K2,R2)))@np.matmul(K2,t2)
    # 2. Compute the new rotation matrix R_
    R_tilda= np.zeros([3,3])
    #r1 = ((c2-c1) /np.linalg.norm(c1-c2)).reshape(-1)
    r1 = ((c2-c1) /np.linalg.norm(c1-c2)).reshape(-1)
    #r2 = np.cross(R1[2],r1)
    r2 = np.array([-r1[1],r1[0],0]) 
    #pdb.set_trace()
    r3 = np.cross(r1,r2)
    R_tilda[0] = r1
    R_tilda[1] = r2
    R_tilda[2] = r3   

    #R1p = R2p = R_tilda
    R1p = R1
    R2p = R_tilda
    #R1p = R_tilda
    #R2p = R2@R_tilda
    K1p = K2p = K2


    t1p = -np.matmul(R_tilda,c1)
    t2p = -np.matmul(R_tilda,c2)

    M1 = np.matmul(K1p,R1p)@ np.linalg.inv(np.matmul(K1,R1))
    M2 = np.matmul(K2p,R2p)@ np.linalg.inv(np.matmul(K2,R2))
    return M1,M2,K1p,K2p,R1p,R2p,t1p,t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    w = (win_size-1)//2
    #im2pad =  np.pad(im2,[(0, 0), (max_disp, max_disp)])
    im1pad =  np.pad(im1,[(0, 0), (max_disp, max_disp)])
    conv2d = np.ones([win_size,win_size])
    height = im1.shape[0]
    width = im1.shape[1]
    
    compares = np.zeros([height,width,max_disp+1])

    for d in range(max_disp+1):
        #A = (im1 - im2pad[:,max_disp-d:max_disp-d + width])**2
        
        A = (im2 - im1pad[:,max_disp-d:max_disp-d + width])**2     
        compares[:,:,d] = scipy.signal.convolve2d(A,conv2d)[w:w + height,w:w+width]
    
    dispM = np.argmin(compares,axis = 2)
    return dispM

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv((np.matmul(K1,R1)))@np.matmul(K1,t1)
    c2 = -np.linalg.inv((np.matmul(K2,R2)))@np.matmul(K2,t2)
    b = np.linalg.norm(c1-c2)
    f = K1[0,0]
    #pdb.set_trace()

    depthM = np.zeros_like(dispM)
    _dispM = dispM
    _dispM[dispM==0] = 1
    _dispM = 1/_dispM
    depthM = b * f * _dispM
    depthM[dispM ==0] = 0

    return depthM

"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    N = x.shape[0]
    A = np.zeros([2*N,12])
    X = np.hstack([X,np.ones([N,1])])
   
    for i in range(N):
        A[2*i] = np.concatenate([X[i],np.zeros([4,]),-x[i,0]*X[i]])
        A[2*i + 1] = np.concatenate([np.zeros([4,]),X[i],-x[i,1]*X[i]])

    _,_,V = np.linalg.svd(A.T@A)
    P = V[-1]
    P = P.reshape([3,4])
    return P

"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    #1. find camera center c
    _,_,V = np.linalg.svd(P.T@P)
    center = V[-1]
    center = (center/center[-1])
    center = (center[:-1]).reshape([-1,1])
    M = P[:,:-1]
    R,K = np.linalg.qr(M)
    t = -M@center   
    #pdb.set_trace()

    return K,R,t
    
