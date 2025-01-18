# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:32:36 2023
I am trying to convert some coordinates geodetic coordinates (WGS84) to ECEF.
@author: mjjimene
"""
import math
import pyproj
import numpy
import rynex7
from pyproj import Proj
from pyproj import CRS
from pyproj import Transformer
import copy


def delete_high_residuals(residuals_network_x,A_P,b_Net_P_x, inc_cal_x):
    dev_res_x =3*numpy.std(residuals_network_x)
    x_top=abs(1*dev_res_x)

    list_a_x = A_P.tolist()

    list_ab_x = b_Net_P_x.tolist()

    for i in reversed(range(len(residuals_network_x))):
        if abs(residuals_network_x[i,0])>x_top:
            list_a_x.pop(i)
            list_ab_x.pop(i)
                      
    dr_x_network= numpy.linalg.lstsq(list_a_x, list_ab_x, rcond=None)[0].flatten().tolist()    

    inc_final_x=inc_cal_x+ float(dr_x_network[0])
    
    return(inc_final_x,list_a_x,list_ab_x)

def matriz_varianza(A_P,x_network,b_Net_P_x,data_geo_coord_network_Base):
    
    res_step1_networkx=numpy.matmul(A_P,x_network)
    res_step_tras_networkx=numpy.matrix(res_step1_networkx).T
    residuals_network_x=numpy.subtract(res_step_tras_networkx, b_Net_P_x)
    rtras_network_x=residuals_network_x.transpose(None)
    RTR_x=numpy.matmul(rtras_network_x,residuals_network_x)
    variance_x =RTR_x/(len(data_geo_coord_network_Base)-1)
    deviation_x0=numpy.sqrt(variance_x)
    a_variance_matrix_network = numpy.array(A_P)
    a_tras_network=a_variance_matrix_network.transpose(None)
    QXX_0_network=a_tras_network.dot(a_variance_matrix_network)
    QXX_network=numpy.linalg.inv(QXX_0_network)
    SigmaXX_x= variance_x* QXX_network[0]
    deviation_x=numpy.sqrt(SigmaXX_x)
    
    return(deviation_x,residuals_network_x)

def ajuste_incrementos(data_geo_coord_network_Base,data_geo_coord_network_Rover,\
                       x_UTM_cal_Net_Base,y_UTM_cal_Net_Base,hae_cal_Net_Base,\
                           x_UTM_cal_Net_R,y_UTM_cal_Net_R,hae_cal_Net_R):
    A=[]
    b_Net_x=[]
    b_Net_y=[]
    b_Net_z=[]
    P_Net=[]

    average_network = 0
    pesos_network = []
    vector_distancias=[]

    for e in range(len(data_geo_coord_network_Base)):
        average_network_Base= (average_network + data_geo_coord_network_Base[e][3])
        varianza_network_Base = average_network_Base/(e+1)
    for e in range(len(data_geo_coord_network_Rover)):
        average_network_Rover= (average_network + data_geo_coord_network_Rover[e][3])
        varianza_network_Rover = average_network_Rover/(e+1)

        varianza=(varianza_network_Base+varianza_network_Rover)/2

    for m in range(len(data_geo_coord_network_Base)):
    # for m in range(len(data_geo_coord_network_Rover)):
    # for m in range(27):
        lat_network_B = data_geo_coord_network_Base[m][0]
        lon_network_B = data_geo_coord_network_Base[m][1]
        hae_network_B = data_geo_coord_network_Base[m][2]
        weigh_network_B = varianza/(data_geo_coord_network_Base[m][3])
    
        lat_network_R = data_geo_coord_network_Rover[m][0]
        lon_network_R = data_geo_coord_network_Rover[m][1]
        hae_network_R = data_geo_coord_network_Rover[m][2]
        weigh_network_R = varianza/(data_geo_coord_network_Rover[m][3])
    
        weigh_network=numpy.sqrt(weigh_network_B*weigh_network_B+weigh_network_R*weigh_network_R)
    
    
        x_UTM_network_B, y_UTM_network_B= rynex7.geod_to_UTM(lat_network_B,lon_network_B,hae_network_B)
        x_UTM_network_R, y_UTM_network_R= rynex7.geod_to_UTM(lat_network_R,lon_network_R,hae_network_R)

        A.append ([1])
    # A.append ([0,1,0])
    # A.append ([0,0,1])
    
        b_Net_x.append([(x_UTM_network_B-x_UTM_network_R)-(x_UTM_cal_Net_Base-x_UTM_cal_Net_R)])
        b_Net_y.append([(y_UTM_network_B-y_UTM_network_R)-(y_UTM_cal_Net_Base-y_UTM_cal_Net_R)])
        b_Net_z.append([(hae_network_B-hae_network_R)-(hae_cal_Net_Base-hae_cal_Net_R)])
        
        sqrt_weigh_network=numpy.sqrt(weigh_network)
        pesos_network.append(sqrt_weigh_network)
    # pesos_network.append(sqrt_weigh_network)
    # pesos_network.append(sqrt_weigh_network)
        diag_weigh_network=numpy.diag(pesos_network)
  
        A_P=numpy.matmul(diag_weigh_network,A)
        b_Net_P_x=numpy.matmul(diag_weigh_network,b_Net_x)
        b_Net_P_y=numpy.matmul(diag_weigh_network,b_Net_y)
        b_Net_P_z=numpy.matmul(diag_weigh_network,b_Net_z)
    
    x_network= numpy.linalg.lstsq(A_P, b_Net_P_x, rcond=None)[0].flatten().tolist() 
    y_network= numpy.linalg.lstsq(A_P, b_Net_P_y, rcond=None)[0].flatten().tolist() 
    z_network= numpy.linalg.lstsq(A_P, b_Net_P_z, rcond=None)[0].flatten().tolist() 
    
    inc_cal_x=float((x_UTM_cal_Net_Base-x_UTM_cal_Net_R))
    inc_cal_y=float((y_UTM_cal_Net_Base-y_UTM_cal_Net_R))
    inc_cal_z=float((hae_cal_Net_Base-hae_cal_Net_R))
    
    inc_final_x=inc_cal_x+ float(x_network[0])
    inc_final_y=inc_cal_y+float(y_network[0])
    inc_final_z=inc_cal_z+float(z_network[0])
        
    # distancia_B_R=numpy.sqrt(inc_final_x**2+inc_final_y**2+inc_final_z**2)
    distancia_B_R=numpy.sqrt(inc_final_x**2+inc_final_y**2)
    # vector_distancias.append(distancia_B_R)
    
    deviation_x,residuals_network_x=rynex7.matriz_varianza(A_P,x_network,b_Net_P_x,data_geo_coord_network_Base)
    deviation_y,residuals_network_y=rynex7.matriz_varianza(A_P,y_network,b_Net_P_y,data_geo_coord_network_Base)
    deviation_z,residuals_network_z=rynex7.matriz_varianza(A_P,z_network,b_Net_P_z,data_geo_coord_network_Base)
    
    return(inc_final_x, inc_final_y, inc_final_z,distancia_B_R,deviation_x,deviation_y,deviation_z,\
           A_P,residuals_network_x,residuals_network_y,residuals_network_z,b_Net_P_x,b_Net_P_y,b_Net_P_z)


def geod_to_geocentric(lat,lon,hae):
    
    transformer = pyproj.Transformer.from_crs(
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},)
    x , y, z = transformer.transform(lon,lat,hae,radians = False)
    
    return(x, y, z)


def geocentric_to_geod(x,y,z):
# x=4929491.886396042
# y=-29024.67857955849
# z=4033761.737604565    
    transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},)
    lon,lat,hae = transformer.transform(x , y, z,radians = False)
    
    return(lon,lat,hae)    

def geod_to_UTM(lat,lon,hae):
# lat= 39.481556
# lon=-0.337396
# # hae=77.967

    trans_to_utm =pyproj.Transformer.from_crs(
    "epsg:4326",
    "+proj=utm +zone=30 +ellps=WGS84",
    always_xy=True,
    )
    xx_UTM, yy_UTM = trans_to_utm.transform(lon, lat)

    return(xx_UTM,yy_UTM)

def UTM_to_geo(x_UTM,y_UTM):
    
    posx, posy = x_UTM, y_UTM

    from_crs = CRS.from_proj4("+proj=utm +zone=30 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    to_crs = CRS.from_epsg(4326)

    proj = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    coordinates = proj.transform(posx, posy)

    return(coordinates)

def SPP_5G_eq(x_m_cal,y_m_cal,z_m_cal,data_geod_coor,data_BS_coor,clk_rec_err_5G):
    
    # making a function to add the 4/5G equations to the initial GNSS model

    # pseudodistances between Nexus's antenna and point 3 on the top of the building
    # (not using iterative process of the before python code)
    # x_azotea, y_azotea, z_azotea = 4929490.578810941, -29023.425326412216, 4033764.626171109 #calculado con los observables GNSS
    # x_azotea, y_azotea, z_azotea = xm_5G, ym_5G, zm_5G #c
    # x_nexus, y_nexus, z_nexus = 4929637.974, -29185.168, 4033575.526 #(observado con estacion total)
    # x_azotea_network, y_azotea_network, z_azotea_network = 4929497.762, -29024.741, 4033748.507
    # x_nexus, y_nexus, z_nexus = 4929653.609, -29185.261, 4033588.406 #(observado con estacion total)
    # x_181, y_181, z_181 = 4929594.106, -28944.530, 4033629.092 #(observado con estacion total)
    # x_930, y_930, z_930 = 4929317.089, -28431.770, 4033984.927 #(proviene de la cartografía, hae la he medido con estacion total)

    aa_eq=[]
    ab_eq=[]
    average = 0
    # p_hae = 0
    weigh_list=[]
    
    for e in range(len(data_geod_coor)):
        average= (average + data_geod_coor[e][3])
        varianza = average/(e+1)
        # print('varianza', varianza)
        
        # (el promedio de todos los valores de azotea 5G)
        # -0.33748433333333333, 39.481261928571435 73.5, 728997,88, 4373569,35
        # print('errors_UTM =', 729010.114- 728997.88, 4373590.076-4373569.35, 72.968-73.5)
    
    for line in range(len(data_geod_coor)):
        # coordenadas del pto 3 calculadas con 5G positioning, son las observadas
        # con ellas calculo
        lat = data_geod_coor[line][0]
        lon = data_geod_coor[line][1]
        hae = data_geod_coor[line][2]
        # divido por 20 el peso porque he comprobado que esa es la proporción de error GNSS/5G para hacer la hibridación 
        # weigh_0 = (varianza/(data_geod_coor[line][3]))
        weigh_0 = (varianza/(data_geod_coor[line][3]))/20
        weigh = (weigh_0/(math.sqrt(1)))
        # print ('lon, lat, hae', lon,lat ,hae)
        # p_hae = (p_hae+ weigh*hae)/(line+1) 

        transformer = pyproj.Transformer.from_crs(
                {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
                {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},)
        x_5G , y_5G , z_5G  = transformer.transform(lon,lat,hae,radians = False)
        # las calculadas provienen del archivo de coordenadas 5G, escogemos la de menor error
        (x_m_cal, y_m_cal, z_m_cal)= (4929484.101236838, -29021.993113274097, 4033758.817417525)

        # (x_azotea_network, y_azotea_network, z_azotea_network)= geod_to_geocentric(lat,lon,hae)
        # print('geocentric_coor', x_azotea_network, y_azotea_network, z_azotea_network)
        for i in range(len(data_BS_coor)):
            x_BS = data_BS_coor[i][0]
            y_BS = data_BS_coor[i][1]
            z_BS = data_BS_coor[i][2]
            
            # print ('x y z BS', x_BS,y_BS,z_BS)
            
            # Array A...
            dx = x_BS - x_m_cal
            dy = y_BS - y_m_cal
            dz = z_BS - z_m_cal

            ro = math.sqrt(
                dx*dx +
                dy*dy +
                dz*dz) #valor calculado
            # print ('ro', ro)

            ax =weigh*(-(x_BS - x_m_cal)/(ro))
            ay =weigh*(-(y_BS - y_m_cal)/(ro))
            az =weigh*(-(z_BS - z_m_cal)/(ro))

            aa_eq.append([ax, ay, az,-1])
            # aa_eq.append([ax, ay, az,0])
            
            # print ('aa equation', ax, ay, az, -1)
    # # Array B... 

            dx_net = x_BS - x_5G 
            dy_net = y_BS - y_5G 
            dz_net = z_BS - z_5G 

            range_obs = math.sqrt(
                dx_net*dx_net +
                dy_net*dy_net+
                dz_net*dz_net) #valor observado
        
            b=weigh*(range_obs - ro+ clk_rec_err_5G)
            ab_eq.append([b])
            # print ('ab equation', b)
      
            weigh_list.append(weigh) 
            
    return(aa_eq, ab_eq,clk_rec_err_5G, weigh_list)

def variance(aa,ab,dr_SPP, weigh_list,lat, lon,hybrid):
    # aa y ab ya están ponderados, no es necesario ponderar los residuos
    deviations_x_y_z_UTM=[]
    
    unos_list=[]
    unos=len(weigh_list)
    for nn in range(unos):
        unos_list.append(1)
    diag_weigh= numpy.diag(unos_list)
    
    # diag_weigh=numpy.diag(weigh_list)
    aa_matrix=numpy.array(aa)
    ab_matrix=numpy.array(ab)
    
    Ax=numpy.matmul(aa_matrix,dr_SPP)
    # ab_matrix=numpy.array(ab_matrix)
    AxT=numpy.matrix(Ax).T
    residuals=numpy.subtract(AxT, ab_matrix)

    # para calcular la varianza de los residuos
    rtras=residuals.transpose(None)
    
    P_Res=diag_weigh.dot(residuals)
    # variance=rtras.dot(residuals)/(len(ab)-4)
    variance=(rtras.dot(P_Res))/(len(ab)-4)
    deviation=math.sqrt(variance)
    
    a_variance_matrix = numpy.array(aa)
    a_tras=a_variance_matrix.transpose(None)
    P_A=diag_weigh.dot(a_variance_matrix)
    QXX_0=a_tras.dot(P_A)
    QXX=numpy.linalg.inv(QXX_0)

    SigmaXX= deviation*deviation*QXX
    deviation_x=math.sqrt(SigmaXX[0][0])
    deviation_y=math.sqrt(SigmaXX[1][1])
    deviation_z=math.sqrt(SigmaXX[2][2])
    # HDOP=(math.sqrt(SigmaXX[0][0]+SigmaXX[1][1]))/deviation
    # VDOP=(math.sqrt(SigmaXX[2][2]))/deviation
    # PDOP=(math.sqrt(SigmaXX[0][0]+SigmaXX[1][1]+SigmaXX[2][2]))/deviation

    Rotation_matrix=rynex7.rotation_matrix(lat,lon)   
    # deviations_x_y_z=deviation_x, deviation_y, deviation_z
    if hybrid==0:

        pru=numpy.delete(SigmaXX, 3, axis=1)
        pru2=numpy.delete(pru, 3, axis=0)
        ro_pru2=numpy.matmul(Rotation_matrix,pru2)
    
    
    # caso hybrid
    if hybrid==1:
        pru=numpy.delete(SigmaXX, 4, axis=1)
        pru1=numpy.delete(pru, 3, axis=1)
        pru2=numpy.delete(pru1, 4, axis=0)
        pru3=numpy.delete(pru2, 3, axis=0)
        ro_pru2=numpy.matmul(Rotation_matrix,pru3)
    
    
    deviation_x_UTM=math.sqrt(abs(ro_pru2[0][0]))
    deviation_y_UTM=math.sqrt(abs(ro_pru2[1][1]))
    deviation_z_UTM=math.sqrt(abs(ro_pru2[2][2]))
    deviations_x_y_z_UTM= deviation_x_UTM,deviation_y_UTM,deviation_z_UTM
    
    HDOP=(deviation_x_UTM+deviation_y_UTM)/deviation
    VDOP=(deviation_z_UTM)/deviation
    PDOP=(deviation_x_UTM+deviation_y_UTM+deviation_z_UTM)/deviation
    
    return(deviation_x, deviation_y, deviation_z,deviation,HDOP,VDOP,PDOP,deviations_x_y_z_UTM)

def coor_calculadas(data_geod_coor_GNSS):
# para calcular el promedio data_geod_coor_GNSS, como valores calculados

    average_network = 0
    weigh_network= 0
    denominador= 0

    for e in range(len(data_geod_coor_GNSS)):
        average_network= (average_network + data_geod_coor_GNSS[e][3])
        varianza_network = average_network/(e+1)
 
    promedio_x_GNSS=0
    promedio_y_GNSS=0
    promedio_z_GNSS=0

# para calcular el promedio ponderado de x, y, hae:
    for p in range (len(data_geod_coor_GNSS)):
        weigh_network = varianza_network/(data_geod_coor_GNSS[p][3])
        promedio_x_GNSS= promedio_x_GNSS+(weigh_network*(data_geod_coor_GNSS[p][0]))
        promedio_y_GNSS= promedio_y_GNSS+(weigh_network*(data_geod_coor_GNSS[p][1]))
        promedio_z_GNSS= promedio_z_GNSS+(weigh_network*(data_geod_coor_GNSS[p][2]))
        
        denominador=denominador+weigh_network

    mean_x_GNSS=promedio_x_GNSS/denominador
    mean_y_GNSS=promedio_y_GNSS/denominador 
    hae_cal_GNSS=promedio_z_GNSS/denominador    

    x_UTM_cal_GNSS, y_UTM_cal_GNSS=geod_to_UTM(mean_x_GNSS,mean_y_GNSS,hae_cal_GNSS)    
    
    return(x_UTM_cal_GNSS, y_UTM_cal_GNSS, hae_cal_GNSS)

def coor_calculadas_mediana(data_geod_coor_GNSS):
# para calcular la mediana de data_geod_coor_GNSS, como valores calculados

    median_value = numpy.median(data_geod_coor_GNSS, axis = 0)

    median_x_GNSS=median_value[0]
    median_y_GNSS=median_value[1]
    hae_cal_GNSS=median_value[2]
    
    x_UTM_cal_GNSS, y_UTM_cal_GNSS=geod_to_UTM(median_x_GNSS,median_y_GNSS,hae_cal_GNSS)    
    
    return(x_UTM_cal_GNSS, y_UTM_cal_GNSS, hae_cal_GNSS)

def adjustment_solution(data_geod_coor_network,x_UTM_cal_network,y_UTM_cal_network,hae_cal_network):
    aa_x_y_z_network=[]
    ab_x_network=[]
    ab_y_network=[]
    ab_z_network=[]

    average_network = 0
    pesos_network = []

    for e in range(len(data_geod_coor_network)):
        average_network= (average_network + data_geod_coor_network[e][3])
        varianza_network = average_network/(e+1)

    for m in range(len(data_geod_coor_network)):
        lat_network = data_geod_coor_network[m][0]
        lon_network = data_geod_coor_network[m][1]
        hae_network = data_geod_coor_network[m][2]
        weigh_network = varianza_network/(data_geod_coor_network[m][3])
    
        x_UTM_network, y_UTM_network= rynex7.geod_to_UTM(lat_network,lon_network,hae_network)
        
        ab_x_network.append([x_UTM_network-x_UTM_cal_network])
        ab_y_network.append([y_UTM_network-y_UTM_cal_network])
        ab_z_network.append([hae_network-hae_cal_network])
        sqrt_weigh=numpy.sqrt(weigh_network)
        pesos_network.append(sqrt_weigh)
        diag_weigh_network=numpy.diag(pesos_network)


    ab_y_network_P=numpy.matmul(diag_weigh_network,ab_y_network)
    ab_z_network_P=numpy.matmul(diag_weigh_network,ab_z_network)
        

    dr_x_network= numpy.linalg.lstsq(aa_x_y_z_network_P, ab_x_network_P, rcond=None)[0].flatten().tolist()    
    dr_y_network= numpy.linalg.lstsq(aa_x_y_z_network_P, ab_y_network_P, rcond=None)[0].flatten().tolist() 
    dr_z_network= numpy.linalg.lstsq(aa_x_y_z_network_P, ab_z_network_P, rcond=None)[0].flatten().tolist() 
    

    x_def=[x_UTM_cal_network]+dr_x_network
    y_def=[y_UTM_cal_network]+dr_y_network
    z_def=[hae_cal_network]+dr_z_network
    x_com=sum(x_def)
    y_com=sum(y_def)
    z_com=sum(z_def)

    return(x_com, y_com, z_com,dr_x_network,dr_y_network,dr_z_network)

# data_geod_coor_network,x_UTM_cal_network,y_UTM_cal_network,hae_cal_network=data_geod_coor_GNSS,x_UTM_cal_GNSS,y_UTM_cal_GNSS,hae_cal_GNSS
def variance_matrix(data_geod_coor_network,x_UTM_cal_network,y_UTM_cal_network,hae_cal_network):
    aa_x_y_z_network=[] #matriz A común a las variables, x,y,hae
    ab_x_network=[] #vector b de la variable x
    ab_y_network=[] #vector b de la variable y
    ab_z_network=[] #vector b de la variable hae

    average_network = 0
    pesos_network = []
# para calcular el peso de cada observable
    for e in range(len(data_geod_coor_network)):
        average_network= (average_network + data_geod_coor_network[e][3])
        varianza_network = average_network/(e+1)

    for m in range(len(data_geod_coor_network)):
        lat_network = data_geod_coor_network[m][0]
        lon_network = data_geod_coor_network[m][1]
        hae_network = data_geod_coor_network[m][2]
        weigh_network = varianza_network/(data_geod_coor_network[m][3])
    
        x_UTM_network, y_UTM_network= rynex7.geod_to_UTM(lat_network,lon_network,hae_network)
        aa_x_y_z_network.append ([1])
        ab_x_network.append([x_UTM_network-x_UTM_cal_network])
        ab_y_network.append([y_UTM_network-y_UTM_cal_network])
        ab_z_network.append([hae_network-hae_cal_network])
        sqrt_weigh=numpy.sqrt(weigh_network)
        # sqrt_weigh=1
        pesos_network.append(sqrt_weigh)
        diag_weigh_network=numpy.diag(pesos_network)

    aa_x_y_z_network_P=numpy.matmul(diag_weigh_network,aa_x_y_z_network)
    ab_x_network_P=numpy.matmul(diag_weigh_network,ab_x_network)
    ab_y_network_P=numpy.matmul(diag_weigh_network,ab_y_network)
    ab_z_network_P=numpy.matmul(diag_weigh_network,ab_z_network)
        
# calculamos las variables de los tres ajustes, para la x, para la y y para la hae
    dr_x_network= numpy.linalg.lstsq(aa_x_y_z_network_P, ab_x_network_P, rcond=None)[0].flatten().tolist()    
    dr_y_network= numpy.linalg.lstsq(aa_x_y_z_network_P, ab_y_network_P, rcond=None)[0].flatten().tolist() 
    dr_z_network= numpy.linalg.lstsq(aa_x_y_z_network_P, ab_z_network_P, rcond=None)[0].flatten().tolist() 
    
        
    res_step1_networkx=numpy.matmul(aa_x_y_z_network,dr_x_network)
    res_step_tras_networkx=numpy.matrix(res_step1_networkx).T
    
    res_step1_networky=numpy.matmul(aa_x_y_z_network,dr_y_network)
    res_step_tras_networky=numpy.matrix(res_step1_networky).T
    
    res_step1_networkz=numpy.matmul(aa_x_y_z_network,dr_z_network)
    res_step_tras_networkz=numpy.matrix(res_step1_networkz).T
    
    residuals_network_x=numpy.subtract(res_step_tras_networkx, ab_x_network)
    residuals_network_y=numpy.subtract(res_step_tras_networky, ab_y_network)
    residuals_network_z=numpy.subtract(res_step_tras_networkz, ab_z_network)

       # para calcular la varianza de los residuos

    rtras_network_x=residuals_network_x.transpose(None)
    rtras_network_y=residuals_network_y.transpose(None)
    rtras_network_z=residuals_network_z.transpose(None)
    
    # aa y ab ya están ponderados, no es necesario ponderar los residuos
    # P_Res_x=diag_weigh_network.dot(residuals_network_x)
    # P_Res_y=diag_weigh_network.dot(residuals_network_y)
    # P_Res_z=diag_weigh_network.dot(residuals_network_z)
   
    # variance_x=rtras_network_x.dot(P_Res_x)/(len(ab_x_network)-1)
    # variance_y=rtras_network_y.dot(P_Res_y)/(len(ab_y_network)-1)
    # variance_z=rtras_network_z.dot(P_Res_z)/(len(ab_z_network)-1)
    
    variance_x=rtras_network_x.dot(residuals_network_x)/(len(ab_x_network)-1)
    variance_y=rtras_network_y.dot(residuals_network_y)/(len(ab_y_network)-1)
    variance_z=rtras_network_z.dot(residuals_network_z)/(len(ab_z_network)-1)
    # variance_x=rtras_network_x.dot(residuals_network_x)/(19)
    # variance_y=rtras_network_y.dot(residuals_network_y)/(19)
    # variance_z=rtras_network_z.dot(residuals_network_z)/(19)
    
    deviation_x0=math.sqrt(variance_x)
    deviation_y0=math.sqrt(variance_y)
    deviation_z0=math.sqrt(variance_z)

    a_variance_matrix_network = numpy.array(aa_x_y_z_network)
    a_tras_network=a_variance_matrix_network.transpose(None)
    # P_A=diag_weigh_network.dot(a_variance_matrix_network)
    # QXX_0_network=a_tras_network.dot(P_A)
    QXX_0_network=a_tras_network.dot(a_variance_matrix_network)
    QXX_network=numpy.linalg.inv(QXX_0_network)
    SigmaXX_x= deviation_x0*deviation_x0* QXX_network
    SigmaXX_y= deviation_y0*deviation_y0* QXX_network
    SigmaXX_z= deviation_z0*deviation_z0* QXX_network
    
    deviation_x=math.sqrt(SigmaXX_x)
    deviation_y=math.sqrt(SigmaXX_y)
    deviation_z=math.sqrt(SigmaXX_z)
    

    # (x_com_res, y_com_res, z_com_res,deviation_x_res, deviation_y_res, deviation_z_res)=rynex7.delete_high_residuals(residuals_network_x,residuals_network_y\
    #                                                             ,residuals_network_z,\
    # ab_x_network_P,ab_y_network_P,ab_z_network_P,aa_x_y_z_network_P,x_UTM_cal_network,y_UTM_cal_network,hae_cal_network)
    
    # para obtener todos los parámetros que necesita la función delete_high_residuals
    return(deviation_x, deviation_y, deviation_z,residuals_network_x, residuals_network_y, residuals_network_z,\
        ab_x_network_P,ab_y_network_P,ab_z_network_P,aa_x_y_z_network_P)
    # return(deviation_x, deviation_y, deviation_z)

# residuals_network_x,residuals_network_y,residuals_network_z,\
#     ab_x_network_P,ab_y_network_P,ab_z_network_P,aa_x_y_z_network_P,x_UTM_cal_network,y_UTM_cal_network,hae_cal_network=residuals_rinex_x, residuals_rinex_y, residuals_rinex_z,\
#     ab_x_rinex_P,ab_y_rinex_P,ab_z_rinex_P,aa_x_y_z_rinex_P,x_UTM_cal_GNSS_rinex,y_UTM_cal_GNSS_rinex,hae_cal_GNSS_rinex
           
            
def rot_matrix(angle):
    Rot_matrix=numpy.ones((2,2))

    Rot_matrix[0][0]=math.cos(angle)
    Rot_matrix[0][1]=math.sin(angle)

    Rot_matrix[1][0]=-math.sin(angle)
    Rot_matrix[1][1]=math.cos(angle)

   
    return (Rot_matrix)           
            
            
            
            
            
            