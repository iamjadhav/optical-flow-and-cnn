import numpy as np
import cv2

#out = cv2.VideoWriter('Problem_1.1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
#out_1 = cv2.VideoWriter('Problem_1.2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
#out_2 = cv2.VideoWriter('Problem_1.2.2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,480))

cap = cv2.VideoCapture("Cars on Highway.mp4")

#reading the video
ret, frame_1 = cap.read()
frame_1 = cv2.resize(frame_1,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA)

previous_frame = cv2.cvtColor(frame_1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame_1)
hsv[:,:,1] = 255
  

def motion_vector(frame_2):
    
    #motion vectors spaced 25x25 from each other
    for i in range(frame_2.shape[0]):        
        for j in range(frame_2.shape[1]):    
            if i % 25 == 0 and j % 25 == 0:  
                arrow_start = (j,i)               
                arrow_end = (i+(2.5 * magnitude[i,j] * np.sin(angle[i,j])), j+(2.5 * magnitude[i,j] * np.cos(angle[i,j]))) 
                #draw the arrows for vectors from start to end point
                frame_2 = cv2.arrowedLine(frame_2, arrow_start, (int(arrow_end[1]),int(arrow_end[0])),(0,0,255), 2)
                #window = cv2.arrowedLine(window, arrow_start, (int(arrow_end[1]),int(arrow_end[0])),(0,0,255), 2)
    return frame_2


while True:
    
    ret_1, frame_2 = cap.read()
    if frame_2 is None:
        break
    
    frame_2 = cv2.resize(frame_2,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA)
    
    #blank canvas to plot the motion vectors
    window = np.zeros_like(frame_2)          
    window_1 = np.zeros_like(frame_2)
    
    next_frame = cv2.cvtColor(frame_2,cv2.COLOR_BGR2GRAY)
    
    #Farnback optical flow algorithm providing the gradients u and v
    optical = cv2.calcOpticalFlowFarneback(previous_frame,next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    optical_1 = optical.copy()
    
    gradient = 2 * (optical_1[:,:,0])
    
    where = np.where(gradient > 1)
    
    for i in range(len(where[0])):
        window_1[where[0][i],where[1][i],:] = frame_2[where[0][i],where[1][i],:]
    cv2.imshow("Cars w/o Background",window_1)   
    #out_2.write(window_1)
    
    #calculating the magnitude and direction of the gradients
    magnitude, angle = cv2.cartToPolar(optical[:,:,0], optical[:,:,1])
    
    #motion vector function to draw the vectors
    frame_2 = motion_vector(frame_2)
    
    #direction put into the hus channel
    hsv[:,:,0] = angle * 180 / np.pi / 2
    
    #magnitude into the value channel 
    hsv[:,:,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
    
    frame_BGR = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV --> BGR ',frame_BGR)
    
    #out_1.write(frame_BGR)
    cv2.imshow('Main Frame',frame_2)
    #out.write(frame_2)
    k = cv2.waitKey(1) & 0xff
    if k == (27) :
        break 
    previous_frame = next_frame
    
#out.release()
#out_1.release() 
#out_2.release()    
cv2.destroyAllWindows()