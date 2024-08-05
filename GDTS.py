import numpy as np
import pandas as pd 
from TransmitterFrame import TransmitterFrameField

import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model  # type: ignore
import joblib 



class GDTS:
    def __init__(self , turn_rate , turn_radius):
        self.airspeed = turn_rate * turn_radius 
        self.turn_rate = np.degrees(turn_rate)
        self.turn_radius = turn_radius 
        self.transm_loc = np.array([3,1,2])
        self.trans_ori = [45 , 45 , 45]

        self.Initialization() 

    def Initialization(self): 
        x , y = map(float , input("Enter the Initial Coordinates of Receiver (coordinates seperated by space) :").split()) 
        self.uav_pos = np.array([x , y , self.transm_loc[2]])
        self.heading = 0
        self.sensing_time = 0.5 # seconds
        self.rec_ori = [0 , 0, self.heading]  

        self.loop_start = self.uav_pos
        self.curr_strength = [] 
        self.strengths = [] 

        self.timestep = 0 
        self.loop = 1
        
        self.magneticFieldComponents() 
        self.curr_strength.append(self.Field)  
        self.strengths.append(self.Field)

        self.total_path_coordinates_x = []
        self.total_path_coordinates_y = []  

        self.grad_dir = []

        self.threshold_dist = 6 
        self.delta = 2  

        self.IterLoop()


    def magneticFieldComponents(self): 

        roll_t = (np.pi/180)*self.trans_ori[0]  # Roll angle of transmitter 
        pitch_t = (np.pi/180)*self.trans_ori[1]  # Pitch angle of transmitter 
        yaw_t = (np.pi/180)*self.trans_ori[2]  # Yaw angle of transmitter 

        R_t_to_i = np.array([
            [np.cos(pitch_t)*np.cos(yaw_t), np.sin(roll_t)*np.sin(pitch_t)*np.cos(yaw_t) - np.cos(roll_t)*np.sin(yaw_t), np.cos(roll_t)*np.sin(pitch_t)*np.cos(yaw_t) + np.sin(roll_t)*np.sin(yaw_t)],
            [np.cos(pitch_t)*np.sin(yaw_t), np.sin(roll_t)*np.sin(pitch_t)*np.sin(yaw_t) + np.cos(roll_t)*np.cos(yaw_t), np.cos(roll_t)*np.sin(pitch_t)*np.sin(yaw_t) - np.sin(roll_t)*np.cos(yaw_t)],
            [-np.sin(pitch_t), np.sin(roll_t)*np.cos(pitch_t), np.cos(roll_t)*np.cos(pitch_t)]
        ])

        roll_r = (np.pi/180)*self.rec_ori[0]  # Roll angle of Reciever 
        pitch_r = (np.pi/180)*self.rec_ori[1]  # Pitch angle of Reciever
        yaw_r = (np.pi/180)*self.rec_ori[2]   # Yaw angle of Reciever

        R_r_to_i = np.array([
            [np.cos(pitch_r)*np.cos(yaw_r), np.sin(roll_r)*np.sin(pitch_r)*np.cos(yaw_r) - np.cos(roll_r)*np.sin(yaw_r), np.cos(roll_r)*np.sin(pitch_r)*np.cos(yaw_r) + np.sin(roll_r)*np.sin(yaw_r)],
            [np.cos(pitch_r)*np.sin(yaw_r), np.sin(roll_r)*np.sin(pitch_r)*np.sin(yaw_r) + np.cos(roll_r)*np.cos(yaw_r), np.cos(roll_r)*np.sin(pitch_r)*np.sin(yaw_r) - np.sin(roll_r)*np.cos(yaw_r)],
            [-np.sin(pitch_r), np.sin(roll_r)*np.cos(pitch_r), np.cos(roll_r)*np.cos(pitch_r)]
        ])

        R_i_to_t = np.transpose(R_t_to_i)
        r = self.uav_pos - self.transm_loc 
        r = R_i_to_t @ r.reshape(-1,1)
        A = np.array([2*r[0]**2 - r[1]**2 - r[2]**2 , 3*r[0]*r[1] ,3*r[0]*r[2] ]).reshape(-1,1) 
        Am = R_t_to_i@ A
        rd = np.linalg.norm(r)
        H = (1/(4*np.pi*(rd**5)))*(Am.reshape(-1,1))
        R_i_to_r = np.transpose(R_r_to_i)
        Hb = R_i_to_r @ H
        
        self.Field = Hb.reshape(3)   

    def predictionsDL(self):
        model = load_model("Models/SourceLocalModel2.h5")
        with open("Models/transformer.pkl" , 'rb') as file :
            transformer = joblib.load(file)  

        data_dict = {
        "rec_x" : self.uav_pos[0] , 
        "rec_y" : self.uav_pos[1] ,
        "rec_z" : self.uav_pos[2] ,
        "roll_r" : self.rec_ori[0] , 
        "pitch_r" : self.rec_ori[1] , 
        "yaw_r" : self.rec_ori[2] ,  
        "mag_x" : self.curr_strength[0],
        "mag_y" : self.curr_strength[1], 
        "mag_z" : self.curr_strength[2],
        }

        data = pd.DataFrame([data_dict])  
        transformed = transformer.transform(data)
        
        prediction = model.predict(transformed)
        predictions = prediction[0][:2] 

        print(prediction[0][:2])

        return predictions


    def IterLoop(self):  
        curr_dist = (self.transm_loc[0]-self.uav_pos[0])**2 + (self.transm_loc[1]-self.uav_pos[0])**2

        x, y = self.uav_pos[0] , self.uav_pos[1] 

        self.magneticFieldComponents()

        self.total_path_coordinates_x.append(x)
        self.total_path_coordinates_y.append(y)

        count = 1 

        while curr_dist > self.threshold_dist: 
            self.timestep += 1

            if count > 8000 :
                break 

            count += 1 

            # Update heading
            self.heading = self.heading + (self.turn_rate * self.sensing_time) 

            if self.heading > 180:
                self.heading -= 360
            elif self.heading < -180:
                self.heading += 360

            self.x_vel = self.airspeed * np.cos(np.radians(self.heading))
            self.y_vel = self.airspeed * np.sin(np.radians(self.heading))

            x += (self.x_vel * self.sensing_time)
            y += (self.y_vel * self.sensing_time)

            self.uav_pos = np.array([x , y , self.transm_loc[2]])

            self.total_path_coordinates_x.append(x)
            self.total_path_coordinates_y.append(y)
            
            self.rec_ori[2] = self.heading 
            self.magneticFieldComponents()

            self.strengths.append(self.Field) 
            self.curr_strength = self.Field  
            self.uav_pos = np.array([x, y , self.transm_loc[-1]])

            self.p = np.array([x, y]) - np.array(self.loop_start[:2])

            print(f"Timestep: {self.timestep}, Heading: {self.heading}, Turn Rate: {self.turn_rate}, Current Position: ({x}, {y})")

            if self.loop == 1:
                if self.timestep > 3 and (np.linalg.norm(self.strengths[self.timestep - 3]) < np.linalg.norm(self.strengths[self.timestep - 2]) and np.linalg.norm(self.strengths[self.timestep - 2]) == np.linalg.norm(self.strengths[self.timestep - 1]) and np.linalg.norm(self.strengths[self.timestep - 1]) > np.linalg.norm(self.strengths[self.timestep])):
                    self.turn_rate = -self.turn_rate
                    self.grad_dir = self.predictionsDL() # Changes 
                    self.strengths = []
                    self.loop_start = self.uav_pos  
                    self.timestep = 0
                    self.loop += 1
                    print("Turn rate changed due to condition 1")

                    continue

                if self.timestep > 2 and (np.linalg.norm(self.strengths[self.timestep - 2]) < np.linalg.norm(self.strengths[self.timestep - 1]) and np.linalg.norm(self.strengths[self.timestep - 1]) > np.linalg.norm(self.strengths[self.timestep])):
                    self.turn_rate = - self.turn_rate
                    self.grad_dir = self.predictionsDL() # changes 
                    self.strengths = []
                    self.loop_start = self.uav_pos   
                    self.timestep = 0
                    self.loop += 1     
                    print("Turn rate changed due to condition 2")

            else:   

                if abs(np.degrees(np.arccos(np.dot(self.p, self.grad_dir) / (np.linalg.norm(self.p) * np.linalg.norm(self.grad_dir))))) < self.delta:
                    self.grad_dir = self.predictionsDL() # changes

                    theta = np.degrees(np.arctan2(self.grad_dir[1], self.grad_dir[0])) 

                    if (self.heading - theta) * self.turn_rate >= 0:
                        self.turn_rate = -self.turn_rate 
                    if np.linalg.norm(self.uav_pos - self.loop_start) <= 0.1: # changes 
                        self.turn_rate = -self.turn_rate
                        print("Turn rate Changed due to Complete Circular Trajectory")

                    self.loop += 1
                    self.timestep = 0
                    self.strengths = []
                    self.loop_start = self.uav_pos 
                    print("Turn rate changed due to gradient direction")   

            curr_dist = (self.transm_loc[0] - x)**2 + (self.transm_loc[1] - y)**2    

        self.plotting()

    def plotting(self):

        x = np.linspace(self.transm_loc[0]-30, self.transm_loc[0]+30 , 100)
        y = np.linspace(self.transm_loc[1]-30, self.transm_loc[1]+30 , 100)
        z = np.full(shape=(100,), fill_value=self.transm_loc[-1])

        X , Y , Z = np.meshgrid(x,y,z)
        
        tffield = TransmitterFrameField([X,Y,Z] , self.transm_loc)

        Bx , By , _ = tffield.B


        plt.figure(figsize=(10,10)) 

        plt.streamplot(X[:,:,0] , Y[:,:,0] ,Bx[:,:,0] , By[:,:,0])  

        plt.plot(self.total_path_coordinates_x , self.total_path_coordinates_y , 'r-')
        plt.scatter([self.total_path_coordinates_x[0]] , [self.total_path_coordinates_y[0]] ,c='m' , s = 100)
        plt.scatter([self.transm_loc[0]] , [self.transm_loc[1]] , c='g' , s=100) 

        plt.xlabel('X-Coordinates')
        plt.ylabel('Y-Coordinates') 

        plt.savefig("GDTS_DL_8.png")

        plt.show()




if __name__ == "__main__":
    turn_rate = float(input("Enter Turn Rate in (rad/sec) : "))
    turn_radius = float(input("Enter Turn Radius in (m) : ")) 

    obj = GDTS(turn_rate , turn_radius)
