from camera_matrix import get_K
import cv2
import numpy as  np
import math

#inputs kp1= key points of reference frame kp2= key points of new frame 
def get_RT(kp2,kp1,K=get_K()):
        # Estimate the essential matrix
        E, mask = cv2.findEssentialMat(kp1,kp2,K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # Estimate Rotation and translation vectors
        _, R, t, mask = cv2.recoverPose(E,kp1,kp2,K)
        return(R,t)


def getRelativeScale(pc_prev,pc_cur):
    """ Returns the relative scale based on the 3-D point clouds
     produced by the triangulation_3D function. Using a pair of 3-D corresponding points
     the distance between them is calculated. This distance is then divided by the
     corresponding points' distance in another point cloud."""

    min_idx = min([pc_cur.shape[0], pc_prev.shape[0]])
    ratios = []  # List to obtain all the ratios of the distances
    for i in range(min_idx):
        if i > 0:
            Xk = pc_cur[i]
            p_Xk = pc_cur[i - 1]
            Xk_1 = pc_prev[i]
            p_Xk_1 = pc_prev[i - 1]

            if np.linalg.norm(p_Xk - Xk) != 0:
                ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

    d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
    return d_ratio








#px_ref is  a key features of which we found previously 
def KLT_featureTracking(prev_img, image_cur, px_ref):
    """Feature tracking using the Kanade-Lucas-Tomasi tracker.
    A backtracking check is done to ensure good features. The backtracking features method
    consist of tracking a set of features, f-1, onto a new frame, which will produce the corresponding features, f-2,
    on the new frame. Once this is done we take the f-2 features, and look for their
    corresponding features, f-1', on the last frame. When we obtain the f-1' features we look for the
    absolute difference between f-1 and f-1', abs(f-1 and f-1'). If the absolute difference is less than a certain
    threshold(in this case 1) then we consider them good features."""

    lk_params = dict(winSize=(21, 21), maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    fMATCHING_DIFF=1


    # Feature Correspondence with Backtracking Check
    kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_img, image_cur, px_ref, None, **lk_params)
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, prev_img, kp2, None, **lk_params)

    d = abs(px_ref - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
    good = d < fMATCHING_DIFF  # Verify which features produced good results by the difference being less
                               # than the fMATCHING_DIFF threshold.
    # Error Management
    if len(d) == 0:
        print('Error: No matches where made.')
    elif list(good).count(
            True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
        print('Warning: No match was good. Returns the list without good point correspondence.')
        return kp1, kp2

    # Create new lists with the good features
    n_kp1, n_kp2 = [], []
    for i, good_flag in enumerate(good):
        if good_flag:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    # Format the features into float32 numpy arrays
    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

    # Verify if the point correspondence points are in the same
    # pixel coordinates. If true the car is stopped (theoretically)
    d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

    # The mean of the differences is used to determine the amount
    # of distance between the pixels
    diff_mean = np.mean(d)

    return n_kp1, n_kp2, diff_mean


def betterMatches(F, points1, points2):
    """ Minimize the geometric error between corresponding image coordinates.
    For more information look into OpenCV's docs for the cv2.correctMatches function."""

    # Reshaping for cv2.correctMatches
    points1 = np.reshape(points1, (1, points1.shape[0], 2))
    points2 = np.reshape(points2, (1, points2.shape[0], 2))

    newPoints1, newPoints2 = cv2.correctMatches(F, points1, points2)

    return newPoints1[0], newPoints2[0]


#Given the rotation matrixi R_{3x3}, compute the rotation angle of each axis, the result is degree format
def rotationMatrix2Angle(R):
        x=math.atan2(R[2][1],R[2][2])/2/math.pi*360.0
        y=math.atan2(-R[2][0],math.sqrt(R[2][1]*R[2][1]+R[2][2]*R[2][2]))/2/math.pi*360.0
        z=math.atan2(R[1][0],R[0][0])/2/math.pi*360.0
        return [x,y,z]






