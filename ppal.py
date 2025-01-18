# -*- coding: utf-8 -*-
"""
Spyder Editor
METODO DE INCREMENTO DE COORDENADAS CON LA NETWORK con SIMPLES DIFERENCIAS
This is a temporary script file.
"""
import numpy
import math
import rynex7

# coordenadas base (banco2) y rover(banco4) de referencia
XB=729063.600
YB=4373540.217
ZB=55.522

XR=729042.486
YR=4373478.627
ZR=55.652

# Coordenadas del rover obtenidos con RTKlib
# XR=729042.523
# YR=4373478.514
# Inc_Z=-0.6537

inc_ref_x=XB-XR
inc_ref_y=YB-YR
inc_ref_z=ZB-ZR
print('inc_ref_x,inc_ref_y,inc_ref_z', inc_ref_x,inc_ref_y,inc_ref_z)

# archivos de datos GNSS y Network de la base y el rover


# dia 17, 
# geo_coord_GNSS_Base = 'c:/RTK_network/GPS_Base_17.txt'
# geo_coord_GNSS_Rover = 'c:/RTK_network/GPS_Rover_17.txt'
# geo_coord_network_Base = 'c:/RTK_network/Base_Network_17.txt'
# geo_coord_network_Rover = 'c:/RTK_network/Rover_Network_17.txt'

# dia 14, sale genial también con el rtklib
geo_coord_GNSS_Base = 'c:/RTK_network/GPS_Base_14.txt'
geo_coord_GNSS_Rover = 'c:/RTK_network/GPS_Rover_14.txt'
geo_coord_network_Base = 'c:/RTK_network/Base_Network_14.txt'
geo_coord_network_Rover = 'c:/RTK_network/Rover_Network_14.txt'

# día 10, ese día el gps tenía problemas por la actividad de la ionosfera
# geo_coord_GNSS_Base = 'c:/RTK_network/GPS_Base_15.txt'
# geo_coord_GNSS_Rover = 'c:/RTK_network/GPS_Rover_15.txt'
# geo_coord_network_Base = 'c:/RTK_network/Base_Network_15.txt'
# geo_coord_network_Rover = 'c:/RTK_network/Rover_Network_15.txt'

# estos datos son muy malos
# geo_coord_GNSS_Base = 'c:/RTK_network/Base_GPS.txt'
# geo_coord_GNSS_Rover = 'c:/RTK_network/Rover_GPS.txt'
# geo_coord_network_Base = 'c:/RTK_network/Base_Network.txt'
# geo_coord_network_Rover = 'c:/RTK_network/Rover_Network.txt'

# estos son los buenos, los del primer dia
# geo_coord_GNSS_Base = 'c:/RTK_network/X2_Base_GPS.txt'
# geo_coord_GNSS_Rover = 'c:/RTK_network/X3_Rover_GPS.txt'
# geo_coord_network_Base = 'c:/RTK_network/Banco2_Network.txt'
# geo_coord_network_Rover = 'c:/RTK_network/Banco4_Network.txt'

data_geo_coord_GNSS_Base= numpy.loadtxt(geo_coord_GNSS_Base)
data_geo_coord_GNSS_Rover= numpy.loadtxt(geo_coord_GNSS_Rover)
data_geo_coord_network_Base = numpy.loadtxt(geo_coord_network_Base)
data_geo_coord_network_Rover= numpy.loadtxt(geo_coord_network_Rover)

# para calcular el promedio data_geod_coor_Network de la Base, como valores calculados 
x_UTM_cal_Net_Base,y_UTM_cal_Net_Base,hae_cal_Net_Base=rynex7.coor_calculadas(data_geo_coord_network_Base)
# print('x_UTM_cal_Net_Base,y_UTM_cal_Net_Base,hae_cal_Net_Base',x_UTM_cal_Net_Base,y_UTM_cal_Net_Base,hae_cal_Net_Base )

# para calcular el promedio data_geod_coor_Network del Rover, como valores calculados 
x_UTM_cal_Net_R,y_UTM_cal_Net_R,hae_cal_Net_R=rynex7.coor_calculadas(data_geo_coord_network_Rover)
# print('x_UTM_cal_Net_R,y_UTM_cal_Net_R,hae_cal_Net_R',x_UTM_cal_Net_R,y_UTM_cal_Net_R,hae_cal_Net_R )

# para calcular el promedio data_geod_coor_GPS de la Base, como valores calculados 
x_UTM_cal_GPS_Base,y_UTM_cal_GPS_Base,hae_cal_GPS_Base=rynex7.coor_calculadas(data_geo_coord_GNSS_Base)
# print('x_UTM_cal_GPS_Base,y_UTM_cal_GPS_Base,hae_cal_GPS_Base',x_UTM_cal_GPS_Base,y_UTM_cal_GPS_Base,hae_cal_GPS_Base)

# para calcular el promedio data_geod_coor_GPS del Rover, como valores calculados 
x_UTM_cal_GPS_R,y_UTM_cal_GPS_R,hae_cal_GPS_R=rynex7.coor_calculadas(data_geo_coord_GNSS_Rover)
# print('x_UTM_cal_GPS_R,y_UTM_cal_GPS_R,hae_cal_GPS_R',x_UTM_cal_GPS_R,y_UTM_cal_GPS_R,hae_cal_GPS_R)

# incrementos calculados Network
inc_cal_x = x_UTM_cal_Net_Base-x_UTM_cal_Net_R
inc_cal_y = y_UTM_cal_Net_Base-y_UTM_cal_Net_R
inc_cal_z = hae_cal_Net_Base-hae_cal_Net_R 
print('inc_cal_x_Net,inc_cal_y_Net,inc_cal_z_Net',inc_cal_x,inc_cal_y,inc_cal_z)

# incrementos calculados GPS
inc_cal_x_GPS = x_UTM_cal_GPS_Base-x_UTM_cal_GPS_R
inc_cal_y_GPS = y_UTM_cal_GPS_Base-y_UTM_cal_GPS_R
inc_cal_z_GPS = hae_cal_GPS_Base-hae_cal_GPS_R 
print('inc_cal_x_GPS,inc_cal_y_GPS,inc_cal_z_GPS',inc_cal_x_GPS,inc_cal_y_GPS,inc_cal_z_GPS)

# incrementos calculados GPS con RTKlib de los datos del día 14 de mayo
inc_RTKlib_x_GPS=22.8961
inc_RTKlib_y_GPS=60.9913
inc_RTKlib_z_GPS=0.6537

# coordenadas del Rover calculadas con RTKlib:
# XR=729042.523
# YR=4373478.514
# Inc_Z=-0.6537

print('día 14:inc_RTKlib_x_GPS,inc_RTKlib_y_GPS,inc_RTKlib_z_GPS',inc_RTKlib_x_GPS,inc_RTKlib_y_GPS,inc_RTKlib_z_GPS) 

promedio_cal_hybrid_x=(inc_cal_x+inc_cal_x_GPS)/2
promedio_cal_hybrid_y=(inc_cal_y+inc_cal_y_GPS)/2

# Para calcular las matrices de diseño, el resultado del ajuste y las desviaciones típicas de los datos NETWORK:
inc_final_x, inc_final_y, inc_final_z,distancia_B_R,deviation_x,deviation_y,deviation_z,\
     A_P,residuals_network_x,residuals_network_y,residuals_network_z,b_Net_P_x,b_Net_P_y,b_Net_P_z\
    =rynex7.ajuste_incrementos(data_geo_coord_network_Base,data_geo_coord_network_Rover,\
                        x_UTM_cal_Net_Base,y_UTM_cal_Net_Base,hae_cal_Net_Base,\
                            x_UTM_cal_Net_R,y_UTM_cal_Net_R,hae_cal_Net_R)

# print('distancia_B_R',distancia_B_R)
# print('deviation_x,deviation_y,deviation_z', deviation_x,deviation_y,deviation_z)

# Para calcular las matrices de diseño, el resultado del ajuste y las desviaciones típicas de los datos GPS:

inc_final_x_G, inc_final_y_G, inc_final_z_G,distancia_B_R_G,deviation_x_G,deviation_y_G,deviation_z_G,\
      A_P_G,residuals_x_G,residuals_y_G,residuals_z_G,b_P_x_G,b_P_y_G,b_P_z_G\
    =rynex7.ajuste_incrementos(data_geo_coord_GNSS_Base,data_geo_coord_GNSS_Rover,\
                        x_UTM_cal_GPS_Base,y_UTM_cal_GPS_Base,hae_cal_GPS_Base,\
                            x_UTM_cal_GPS_R,y_UTM_cal_GPS_R,hae_cal_GPS_R)
# print('distancia_B_R_G',distancia_B_R_G)
# print('deviation_x_G,deviation_y_G,deviation_z_G', deviation_x_G,deviation_y_G,deviation_z_G)

# para eliminar los residuos altos y calcular de nuevo el ajuste de la Network
inc_final_x_delete_residuals_N,list_a_x_N,list_ab_x_N=rynex7.delete_high_residuals(residuals_network_x,A_P,b_Net_P_x, inc_cal_x)
inc_final_y_delete_residuals_N,list_a_y_N,list_ab_y_N=rynex7.delete_high_residuals(residuals_network_y,A_P,b_Net_P_y, inc_cal_y)
inc_final_z_delete_residuals_N,list_a_z_N,list_ab_z_N=rynex7.delete_high_residuals(residuals_network_z,A_P,b_Net_P_z, inc_cal_z)

distancia_B_R_2=numpy.sqrt(inc_final_x_delete_residuals_N**2+inc_final_y_delete_residuals_N**2)

print('inc_final_x_delete_residuals_Net,inc_final_y_delete_residuals_Net,inc_final_z_delete_residuals_N',\
      inc_final_x_delete_residuals_N,inc_final_y_delete_residuals_N,inc_final_z_delete_residuals_N)
print('distancia_B_R_2 eliminando residuos altos Network',distancia_B_R_2)

# para eliminar los residuos altos y calcular de nuevo el ajuste de la red GPS
inc_final_x_delete_residuals_G,list_a_x_G,list_ab_x_G=rynex7.delete_high_residuals(residuals_x_G,A_P_G,b_P_x_G, inc_cal_x_GPS)
inc_final_y_delete_residuals_G,list_a_y_G,list_ab_y_G=rynex7.delete_high_residuals(residuals_y_G,A_P_G,b_P_y_G, inc_cal_y_GPS)
inc_final_z_delete_residuals_G,list_a_z_G,list_ab_z_G=rynex7.delete_high_residuals(residuals_z_G,A_P_G,b_P_z_G, inc_cal_z_GPS)

distancia_B_R_2_G=numpy.sqrt(inc_final_x_delete_residuals_G**2+inc_final_y_delete_residuals_G**2)

print('inc_final_x_delete_residuals_G,inc_final_y_delete_residuals_G,inc_final_z_delete_residuals_G',\
      inc_final_x_delete_residuals_G,inc_final_y_delete_residuals_G,inc_final_z_delete_residuals_G)
print('distancia_B_R_2_G eliminando residuos altos GPS',distancia_B_R_2_G)

real_distance=numpy.sqrt(inc_ref_x**2+inc_ref_y**2)

print('inc_ref_x,inc_ref_y,inc_ref_z',inc_ref_x,inc_ref_y,inc_ref_z)
print('real_distance B-R',real_distance)

# Para calcular las desviaciones típicas después de eliminar los residuos altos Network
 
dr_x_N= numpy.linalg.lstsq(list_a_x_N, list_ab_x_N, rcond=None)[0].flatten().tolist() 
dr_y_N= numpy.linalg.lstsq(list_a_y_N, list_ab_y_N, rcond=None)[0].flatten().tolist()  
dr_z_N= numpy.linalg.lstsq(list_a_z_N, list_ab_z_N, rcond=None)[0].flatten().tolist()       

deviation_x_N_R,residuals_GPS_x=rynex7.matriz_varianza(list_a_x_N,dr_x_N,list_ab_x_N,list_a_x_N)
deviation_y_N_R,residuals_GPS_y=rynex7.matriz_varianza(list_a_y_N,dr_y_N,list_ab_y_N,list_a_y_N)
deviation_z_N_R,residuals_GPS_z=rynex7.matriz_varianza(list_a_z_N,dr_z_N,list_ab_z_N,list_a_z_N)
    
print('deviation_x_N_R,deviation_y_N_R,deviation_z_N_R', deviation_x_N_R,deviation_y_N_R,deviation_z_N_R)

# Para calcular las desviaciones típicas después de eliminar los residuos altos GPS
 
dr_x_GPS= numpy.linalg.lstsq(list_a_x_G, list_ab_x_G, rcond=None)[0].flatten().tolist() 
dr_y_GPS= numpy.linalg.lstsq(list_a_y_G, list_ab_y_G, rcond=None)[0].flatten().tolist()  
dr_z_GPS= numpy.linalg.lstsq(list_a_z_G, list_ab_z_G, rcond=None)[0].flatten().tolist()       

deviation_x_G_R,residuals_GPS_x=rynex7.matriz_varianza(list_a_x_G,dr_x_GPS,list_ab_x_G,list_a_x_G)
deviation_y_G_R,residuals_GPS_y=rynex7.matriz_varianza(list_a_y_G,dr_y_GPS,list_ab_y_G,list_a_y_G)
deviation_z_G_R,residuals_GPS_z=rynex7.matriz_varianza(list_a_z_G,dr_z_GPS,list_ab_z_G,list_a_z_G)
    
print('deviation_x_G_R,deviation_y_G_R,deviation_z_G_R', deviation_x_G_R,deviation_y_G_R,deviation_z_G_R)

# # Cambio de datum: 
#     # calcular angulos a partir de los incrementos de coordenadas GNSS

inc_final_x_GNSS=inc_final_x_delete_residuals_G
inc_final_y_GNSS=inc_final_y_delete_residuals_G

angle_rad_net=math.atan(inc_final_x_delete_residuals_N/inc_final_y_delete_residuals_N)
angle_gra_net = math.degrees(angle_rad_net)
dist_net= distancia_B_R_2

# # angle_rad =math.atan(inc_final_y_GNSS/inc_final_x_GNSS)
# angle_gra = math.degrees(angle_rad) 

if inc_final_x_GNSS>0 and inc_final_y_GNSS>0:
    angle_rad =math.atan(inc_final_x_GNSS/inc_final_y_GNSS)
    inc_x_new= math.sin(angle_rad)*dist_net
    inc_y_new= math.cos(angle_rad)*dist_net
    
if inc_final_x_GNSS>0 and inc_final_y_GNSS<0:
    angle_rad =math.atan(inc_final_y_GNSS/inc_final_x_GNSS)
    inc_x_new= math.cos(angle_rad)*dist_net
    inc_y_new= math.sin(angle_rad)*dist_net

if inc_final_x_GNSS<0 and inc_final_y_GNSS<0:
    angle_rad =math.atan(inc_final_x_GNSS/inc_final_y_GNSS)
    # azimut= 1/angle_rad +(math.pi)
    inc_x_new=- math.cos(angle_rad)*dist_net
    inc_y_new= math.sin(angle_rad)*dist_net
    
if inc_final_x_GNSS<0 and inc_final_y_GNSS>0:
    angle_rad =math.atan(inc_final_y_GNSS/inc_final_x_GNSS)
    inc_x_new= math.cos(angle_rad)*dist_net
    inc_y_new= math.sin(angle_rad)*dist_net
    
# comparamos con la referencia los nuevos incrementos mejorados GPS
accuracy_x=inc_ref_x-inc_x_new
accuracy_y=inc_ref_y-inc_y_new

# comparamos con la referencia los  incrementos calculados GPS
accuracy_x_initial=inc_ref_x-inc_cal_x_GPS
accuracy_y_initial=inc_ref_y-inc_cal_y_GPS

# print('accuracy_x,accuracy_y',accuracy_x,accuracy_y)
# print('accuracy_x_cal,accuracy_y_cal',accuracy_x_initial,accuracy_y_initial)

# Para calcular con la matriz rotación los incrementos de la network en el datum GPS
# es un segundo método para transformar un datum en otro, y el resultado es correcto
angle_gra = math.degrees(angle_rad) 

# rot es el ángulo de rotación para cambiar de datum a los incrementos network
rot=angle_rad_net-angle_rad 

Rotation_matrix=rynex7.rot_matrix(rot)

[inc_rot_x, inc_rot_y]=numpy.matmul([inc_final_x_delete_residuals_N,inc_final_y_delete_residuals_N],Rotation_matrix)

print('inc_rot_x_net, inc_rot_y_net,inc_final_z_net',inc_rot_x, inc_rot_y,inc_final_z_delete_residuals_N)

# Como ya hemos cambiado de datum podemos utilizar gps y network para calcular los incrementos
inc_x_hybrid=(inc_rot_x+inc_final_x_GNSS)/2
inc_y_hybrid=(inc_rot_y+inc_final_y_GNSS)/2
# inc_z_hybrid=inc_final_z_delete_residuals_G
inc_z_hybrid=inc_final_z_delete_residuals_N
# inc_z_hybrid=(inc_final_z_delete_residuals_G+inc_final_z_delete_residuals_N)/2
print('inc_x_hybrid, inc_y_hybrid,inc_z_Net',inc_x_hybrid, inc_y_hybrid,inc_z_hybrid)

accuracy_x_hybrid=inc_ref_x-inc_x_hybrid
accuracy_y_hybrid=inc_ref_y-inc_y_hybrid
accuracy_z_hybrid=inc_ref_z-inc_z_hybrid

print('accuracy_x_hybrid, accuracy_y_hybrid,accuracy_z_NET',accuracy_x_hybrid, accuracy_y_hybrid,accuracy_z_hybrid)