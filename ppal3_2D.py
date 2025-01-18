# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:44:13 2024
DOBLES DIFERENCIAS (PSEUDODISTANCIAS) CON LA NETWORK
@author: mjjimene
"""

import numpy
import math
import rynex7 #function geod_to_geocentric and others

# coordenadas base (banco2) y rover(banco4) de referencia
XB=729063.600
YB=4373540.217
ZB=55.522

XR=729042.486
YR=4373478.627
ZR=55.652

inc_ref_x=XB-XR
inc_ref_y=YB-YR
inc_ref_z=ZB-ZR
print('inc_ref_x,inc_ref_y,inc_ref_z', inc_ref_x,inc_ref_y,inc_ref_z)

# Coordenadas de las 3 Base Stations
Base_Stations_coord = 'c:/RTK_network/coor_Base_Stations.txt'
data_BS_coor = numpy.loadtxt(Base_Stations_coord)

# archivos de datos GNSS y Network de la base y el rover

# día 10, ese día el gps tenía problemas por la actividad de la ionosfera
# geo_coord_GNSS_Base = 'c:/RTK_network/GPS_Base_15.txt'
# geo_coord_GNSS_Rover = 'c:/RTK_network/GPS_Rover_15.txt'
# geo_coord_network_Base = 'c:/RTK_network/Base_Network_15.txt'
# geo_coord_network_Rover = 'c:/RTK_network/Rover_Network_15.txt'

geo_coord_GNSS_Base = 'c:/RTK_network/GPS_Base_14.txt'
geo_coord_GNSS_Rover = 'c:/RTK_network/GPS_Rover_14.txt'
geo_coord_network_Base = 'c:/RTK_network/Base_Network_14.txt'
geo_coord_network_Rover = 'c:/RTK_network/Rover_Network_14.txt'

data_geo_coord_GNSS_Base= numpy.loadtxt(geo_coord_GNSS_Base)
data_geo_coord_GNSS_Rover= numpy.loadtxt(geo_coord_GNSS_Rover)
data_geo_coord_network_Base = numpy.loadtxt(geo_coord_network_Base)
data_geo_coord_network_Rover= numpy.loadtxt(geo_coord_network_Rover)

# para calcular el promedio data_geod_coor_Network de la Base, como valores calculados 
x_UTM_cal_Net_Base,y_UTM_cal_Net_Base,z_UTM_cal_Net_Base=rynex7.coor_calculadas(data_geo_coord_network_Base)

# para calcular el promedio data_geod_coor_Network del Rover, como valores calculados 
x_UTM_cal_Net_R,y_UTM_cal_Net_R,z_UTM_cal_Net_R=rynex7.coor_calculadas(data_geo_coord_network_Rover)
x_UTM_R_CAL=x_UTM_cal_Net_R
y_UTM_R_CAL=y_UTM_cal_Net_R
z_UTM_R_CAL=z_UTM_cal_Net_R
# para calcular el promedio data_geod_coor_GPS de la Base, como valores calculados 
x_UTM_cal_GPS_Base,y_UTM_cal_GPS_Base,hae_cal_GPS_Base=rynex7.coor_calculadas(data_geo_coord_GNSS_Base)

# para calcular el promedio data_geod_coor_GPS del Rover, como valores calculados 
x_UTM_cal_GPS_R,y_UTM_cal_GPS_R,hae_cal_GPS_R=rynex7.coor_calculadas(data_geo_coord_GNSS_Rover)

clk_rec_err_5G=0
# 5G solution
#  matrices de diseño del Rover

# para calcular la varianza y despues definir la matriz de pesos
average_network = 0
pesos_network = []

for e in range(len(data_geo_coord_network_Base)):
    average_network_Base= (average_network + data_geo_coord_network_Base[e][3])
    varianza_network_Base = average_network_Base/(e+1)
for e in range(len(data_geo_coord_network_Rover)):
    average_network_Rover= (average_network + data_geo_coord_network_Rover[e][3])
    varianza_network_Rover = average_network_Rover/(e+1)
    varianza=(varianza_network_Base+varianza_network_Rover)/2

it= 0

while it<1:
 
    vector_b=[]
    matrix_A=[]
    pesos_network=[]
    
    for i in range(len(data_BS_coor)):
        if i<2:
            q=i+1
        else:
            q=i-2
        ro_BS1_Base=math.sqrt((data_BS_coor[i][0]-x_UTM_cal_Net_Base)**2+(data_BS_coor[i][1]-y_UTM_cal_Net_Base)**2+(data_BS_coor[i][2]-z_UTM_cal_Net_Base)**2)
        ro_BS2_Base=math.sqrt((data_BS_coor[q][0]-x_UTM_cal_Net_Base)**2+(data_BS_coor[q][1]-y_UTM_cal_Net_Base)**2+(data_BS_coor[q][2]-z_UTM_cal_Net_Base)**2)
        ro_BS1_Rover=math.sqrt((data_BS_coor[i][0]-x_UTM_cal_Net_R)**2+(data_BS_coor[i][1]-y_UTM_cal_Net_R)**2+(data_BS_coor[i][2]-z_UTM_cal_Net_R)**2)
        ro_BS2_Rover=math.sqrt((data_BS_coor[q][0]-x_UTM_cal_Net_R)**2+(data_BS_coor[q][1]-y_UTM_cal_Net_R)**2+(data_BS_coor[q][2]-z_UTM_cal_Net_R)**2)
            
        ro_RB1=-ro_BS1_Rover+ro_BS1_Base
        ro_RB2=-ro_BS2_Rover+ro_BS2_Base
        
        x_UTM_BS1=data_BS_coor[i][0]
        y_UTM_BS1=data_BS_coor[i][1]
        z_UTM_BS1=data_BS_coor[i][2]
        x_UTM_BS2=data_BS_coor[q][0]
        y_UTM_BS2=data_BS_coor[q][1]
        z_UTM_BS2=data_BS_coor[q][2]
            
     
        for p in range(len(data_geo_coord_network_Base)):
             x_UTM_B, y_UTM_B=rynex7.geod_to_UTM(data_geo_coord_network_Base[p][0],data_geo_coord_network_Base[p][1],data_geo_coord_network_Base[p][2])
             R_BS1_Base=math.sqrt((data_BS_coor[i][0]-x_UTM_B)**2+(data_BS_coor[i][1]-y_UTM_B)**2)
             R_BS2_Base=math.sqrt((data_BS_coor[q][0]-x_UTM_B)**2+(data_BS_coor[q][1]-y_UTM_B)**2)
             z_UTM_B=data_geo_coord_network_Base[p][2]
             
             x_UTM_R, y_UTM_R=rynex7.geod_to_UTM(data_geo_coord_network_Rover[p][0],data_geo_coord_network_Rover[p][1],data_geo_coord_network_Rover[p][2])
             R_BS1_Rover=math.sqrt((data_BS_coor[i][0]-x_UTM_R)**2+(data_BS_coor[i][1]-y_UTM_R)**2)
             R_BS2_Rover=math.sqrt((data_BS_coor[q][0]-x_UTM_R)**2+(data_BS_coor[q][1]-y_UTM_R)**2)
             
             
             R_RB1=-R_BS1_Rover+R_BS1_Base
             R_RB2=-R_BS2_Rover+R_BS2_Base

             z_UTM_R=data_geo_coord_network_Rover[p][2]
             
             b=(R_BS1_Base-R_BS1_Rover)-(R_BS2_Base-R_BS2_Rover)-(ro_BS1_Base-ro_BS1_Rover)+(ro_BS2_Base-ro_BS2_Rover)
             # b=-R_BS1_Base+R_BS1_Rover-R_BS2_Base+R_BS2_Rover+ro_BS1_Base-ro_BS1_Rover-ro_BS2_Base+ro_BS2_Rover
             vector_b.append([b])
     
    # matriz A
    
             dx_r1_BS = -x_UTM_cal_Net_R + x_UTM_BS1
             dy_r1_BS = -y_UTM_cal_Net_R + y_UTM_BS1
             dz_r1_BS = -z_UTM_cal_Net_R + z_UTM_BS1   

             dx_r2_BS = -x_UTM_cal_Net_R + x_UTM_BS2
             dy_r2_BS = -y_UTM_cal_Net_R + y_UTM_BS2
             dz_r2_BS = -z_UTM_cal_Net_R + z_UTM_BS2                              
    
                   
             ax_s1_m=(-dx_r1_BS/(ro_BS1_Rover))+(dx_r2_BS/(ro_BS2_Rover))
             ay_s1_m=(-dy_r1_BS/(ro_BS2_Rover))+(dy_r2_BS/(ro_BS2_Rover))
             az_s1_m=(-dz_r1_BS/(ro_BS2_Rover))+(dz_r2_BS/(ro_BS2_Rover))
             # A=([(ax_s1_m), (ay_s1_m), (az_s1_m)])
             A=([(ax_s1_m), (ay_s1_m)])
             matrix_A.append(A)
    
    # matriz de Pesos 
             weigh_network_B = varianza/(data_geo_coord_network_Base[p][3])
             weigh_network_R = varianza/(data_geo_coord_network_Rover[p][3])
             weigh_network=numpy.sqrt(weigh_network_B*weigh_network_B+weigh_network_R*weigh_network_R)
             sqrt_weigh_network=numpy.sqrt(weigh_network)
             pesos_network.append(sqrt_weigh_network)
    
    it=it+1
    
    
    n=len(matrix_A)
    DD_weigh_network=numpy.ones([n,n])
    for r in range(n):
        DD_weigh_network[r][r]=pesos_network[r]
    
    matrix_A_P=numpy.matmul(DD_weigh_network,matrix_A)
    vector_b_P=numpy.matmul(DD_weigh_network,vector_b)
            # 
    dr_SD = numpy.linalg.lstsq(matrix_A, vector_b, rcond=None)[0].flatten().tolist()
    dr_SD_P = numpy.linalg.lstsq(matrix_A_P, vector_b_P, rcond=None)[0].flatten().tolist()
    
    x_comp=x_UTM_cal_Net_R+dr_SD[0]
    y_comp=y_UTM_cal_Net_R+dr_SD[1]
    # z_comp=z_UTM_cal_Net_R+dr_SD[2]
    inc_comp_x=x_UTM_cal_Net_Base-x_comp
    inc_comp_y=y_UTM_cal_Net_Base-y_comp
    # inc_comp_z=z_UTM_cal_Net_Base-z_comp
    
    
    x_comp_P=x_UTM_cal_Net_R+dr_SD_P[0]
    y_comp_P=y_UTM_cal_Net_R+dr_SD_P[1]
    # z_comp_P=z_UTM_cal_Net_R+dr_SD_P[2]
    inc_comp_x_P=x_UTM_cal_Net_Base-x_comp_P
    inc_comp_y_P=y_UTM_cal_Net_Base-y_comp_P
    # inc_comp_z_P=z_UTM_cal_Net_Base-z_comp_P


    print('inc_comp_x_P, inc_comp_y_P,inc_comp_z_P, it',inc_comp_x_P, inc_comp_y_P, it)
    x_UTM_cal_Net_R=x_comp_P
    y_UTM_cal_Net_R=y_comp_P
    # z_UTM_cal_Net_R=z_comp_P
    print('inc_comp_x, inc_comp_y,inc_comp_z, it',inc_comp_x, inc_comp_y, it)
    x_UTM_cal_Base_R=x_comp
    y_UTM_cal_Base_R=y_comp
    # z_UTM_cal_Base_R=z_comp
    


# Calculo de la matriz varianza covarianza

res1=(numpy.matmul(matrix_A_P,dr_SD_P))
res1T=numpy.matrix(res1).T
res2=numpy.subtract(res1T,vector_b_P)

res2_T=res2.transpose(None)
variance=res2_T.dot(res2)/(len(vector_b)-4)
deviation=math.sqrt(variance[0,0])

# a_variance_matrix_network = numpy.array(aa_x_y_z_network)
matrix_A_P_T=matrix_A_P.transpose(None)
QXX_0=matrix_A_P_T.dot(matrix_A_P)
QXX=numpy.linalg.inv(QXX_0)
SigmaXX=numpy.multiply(QXX,variance)
sigx=SigmaXX[0,0]
sigy=SigmaXX[1,1]
# sigz=SigmaXX[2,2]

deviation_x=math.sqrt(sigx)
deviation_y=math.sqrt(sigy)
# deviation_z=math.sqrt(sigz)

print('deviation_x,deviation_y,deviation_z',deviation_x,deviation_y)
# # Matriz rotación para transformar al datum GNSS

angle_rad_net=math.atan(inc_comp_x_P/inc_comp_y_P)
angle_rad =math.atan(18.57/60.91)
angle_gra_net = math.degrees(angle_rad_net)
angle_gra = math.degrees(angle_rad)

rotation_angle=angle_rad_net-angle_rad 

# Rotation_matrix=rynex7.rot_matrix(rotation_angle)
Rot_matrix=numpy.ones((2,2))
Rot_matrix[0][0]=math.cos(rotation_angle)
Rot_matrix[0][1]=math.sin(rotation_angle)
Rot_matrix[1][0]=-math.sin(rotation_angle)
Rot_matrix[1][1]=math.cos(rotation_angle)
[inc_rot_x, inc_rot_y]=numpy.matmul([inc_comp_x_P,inc_comp_y_P],Rot_matrix)

print('inc_rot_x_net, inc_rot_y_net,inc_final_z_net',inc_rot_x, inc_rot_y)

 
    
# mean_res=1*numpy.std(res2)
# top=abs(1*mean_res)

# for i in reversed(range(len(res2))):
#     if abs(res2[i,0])>top:
#             matrix_A.pop(i)
#             vector_b.pop(i)

# dr_SD_res = numpy.linalg.lstsq(matrix_A, vector_b, rcond=None)[0].flatten().tolist()

# x_comp_res=x_UTM_cal_Net_R+dr_SD_res[0]
# y_comp_res=y_UTM_cal_Net_R+dr_SD_res[1]
# z_comp_res=z_UTM_cal_Net_R+dr_SD_res[2]


# inc_comp_x_res=x_UTM_cal_Net_Base-x_comp_res
# inc_comp_y_res=y_UTM_cal_Net_Base-y_comp_res
# inc_comp_z_res=z_UTM_cal_Net_Base-z_comp_res


          