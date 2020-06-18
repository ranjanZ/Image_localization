from core import *
import numpy as np



class localization:
    def __init__(self):
        self.stage=0        #stage 1 for first; 2 for 2nd  frame
        self.new_frame=None       #The current frame
        self.last_frame=None      #The previous frame

        self.cur_R=None
        self.cur_t=None
        self.K=get_K()          #camera matrix
        self.new_pc=None        #3D point cloud (#num,3)
        self.prev_pc=None       #previous 3D point cloud (#num,3)
        self.new_pc=None        #3D point cloud (#num,3)
        self.kp_prev = None     #previous corresponded feature keypoint
        self.kp_cur = None      #current corresponded key point

        self.T_vectors=[]     #List contains all the translation vectors of each  (1 ,3)
        self.R_matrices=[]     #List contains all the Rotation matrices(#,3,3)
        self.Scale=None 
        self.MinF=500

    def getNewFeatures(self,img):
        feature_pts = cv2.xfeatures2d.SIFT_create().detect(img, None)
        feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)
        return(feature_pts)


    def processFirstFrame(self,frame):
        """Process the first frame. Detects feature points on the first frame
        in order to provide them to the Kanade-Lucas-Tomasi Tracker"""
        cur_img=frame
        self.last_frame=frame 
        self.kp_prev =self.getNewFeatures(cur_img) 
        self.cur_R=np.identity(3)
        self.cur_t=np.array([[0], [0], [0]])
        self.T_vectors.append(self.cur_t)
        self.R_matrices.append(self.cur_R)
        self.stage = 2           #enter to second satage

    def process2ndFrame(self,frame):
        """Process the second frame. Detects feature correspondence between the first frame and the second frame with the Kanade-Lucas-Tomasi Tracker. Initializes the  rotation matrix and translation vector. The first point cloud is formulated."""

        self.new_frame=frame
        prev_img, cur_img = self.last_frame, self.new_frame
        # Obtain feature correspondence points
        self.kp_prev, self.kp_cur, _diff = KLT_featureTracking(prev_img, cur_img, self.kp_prev)

        self.cur_R,self.cur_t=get_RT(self.kp_cur,self.kp_prev,self.K)

        translation=self.cur_R.dot(self.cur_t)
        self.T_vectors.append(tuple(translation))
        self.R_matrices.append(tuple(self.cur_R))


        # Triangulation, returns 3-D point cloud assuming 1st camera is at (0,0,0)
        #self.new_pc = self.get_3Dpoint(self.cur_R, self.cur_t)   
        self.new_pc = np.random.random((100,3))   
        

        #update
        self.kp_prev = self.kp_cur
        self.prev_pc = self.new_pc    #first 3D points
        self.last_frame = self.new_frame
        self.stage = 3


    def processDefaultFrame(self,frame):
        """Process the second frame. Detects feature correspondence between the first frame and the second frame with the Kanade-Lucas-Tomasi Tracker. Initializes the  rotation matrix and translation vector. The first point cloud is formulated."""

        self.new_frame=frame
        prev_img, cur_img = self.last_frame, self.new_frame
        # Obtain feature correspondence points
        self.kp_prev, self.kp_cur, diff = KLT_featureTracking(prev_img, cur_img, self.kp_prev)

        # Verify if the current frame is going to be skipped if no movement in car
        #if skipped  return
       
        if(diff<3):
            if(self.kp_prev.shape[0]<self.MinF):
                self.kp_cur =self.getNewFeatures(prev_img)
                self.kp_prev=self.kp_cur
                #self.prev_pc=self.new_pc
                return
        
        #print(self.kp_prev, self.kp_cur)
        #find R and T  by essential matrix
        #print(self.kp_cur[0],self.kp_prev[0],self.K)
        R,t=get_RT(self.kp_cur,self.kp_prev,self.K)
        #print(R,t)
        self.new_pc = np.random.random((100,3))   

        
        if (abs(t[2]) > abs(t[0]) and abs(t[2]) > abs(t[1])):  # Accepts only dominant forward motion
            self.Scale=getRelativeScale(self.prev_pc,self.new_pc)
            #self.cur_t = self.cur_t - self.Scale * self.cur_R.dot(t)  # Concatenate the translation vectors
            self.cur_t = self.cur_t - self.Scale * R.dot(t)  # Concatenate the translation vectors
            self.cur_R = R.dot(self.cur_R)  # Concatenate the rotation matrix
            self.T_vectors.append(tuple(self.cur_t))
            self.R_matrices.append(tuple(self.cur_R))


        if self.kp_prev.shape[0] < self.MinF:                     # Verify if the amount of feature points
            self.kp_cur = self.getNewFeatures(cur_img)  # is above the kMinNumFeature threshold

        # Triangulation, returns 3-D point cloud assuming 1st camera is at (0,0,0)
        #self.new_pc = self.get_3Dpoint(self.cur_R, self.cur_t)   
        self.new_pc = np.random.random((100,3))   
        


        self.last_frame = self.new_frame
        self.kp_prev = self.kp_cur
        self.last_pc = self.new_pc    
        self.stage = 3

    






















#img_loc=localization()
#img_loc.processFirstFrame(img)
