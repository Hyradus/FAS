#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:59:11 2021
s
@author: hyradus
"""
#####################################################  BLOCK ZERO
import itertools
from more_itertools import split_after
import os
import shutil
from datetime import datetime
import pandas as pd
# import time
from shapely.geometry import Point, LineString
import shapely
# from shapely import ops, geometry
from shapely.ops import substring
import geopandas as gpd
# import os
# from argparse import ArgumentParser
from tkinter import Tk,filedialog
from pyproj import CRS as CRS
import rasterio as rio
from rasterio.plot import reshape_as_image#, reshape_as_raster
import numpy as np 
import matplotlib.pyplot as plt
# from rasterstats import zonal_stats as zns
import math
from tqdm import tqdm
from shapely import ops, geometry
# from shapely.geometry import LineString


GCS_Moon_2000 = CRS.from_wkt('GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')

def question(question, answers):
    answ = None
    while answ not in answers:
        print('Please enter only: ')
        print(*answers, sep=', ')
        
        answ = input('\n'+question+'  Answer: ')
    return(answ)

def make_folder(path, name):
    # os.getcwd()
    folder = path+'/'+name
    if os.path.exists(folder):
           qst = name + ' Folder exist, remove it? '
           answ = question(qst,['yes','y','no','n'])
           if answ in ['yes', 'y']:
               shutil.rmtree(folder)
               os.mkdir(folder)
               print(name, 'Folder created')
           else:
               now = datetime.now()
               folder = path+'/'+name +'_' + now.strftime("%d-%m-%Y_%H-%M-%S")
               print(folder, ' Folder not exist, creating.')
               os.mkdir(folder)
               print('Created new ', name,' Folder')
    else:
        print(name, ' Folder not exist, creating.')
        os.mkdir(folder)
        print('Created new ', name,' Folder')
    return(folder)

def get_MP_PX_coords(indx,dataframe_name, affine, steps):
    try:
        dataframe=gpd.read_file(dataframe_name)
    except:
        dataframe = dataframe_name.copy()
    line = dataframe.loc[indx]['geometry']
    mp = shapely.geometry.MultiPoint()
    for i in np.arange(0, line.length, steps):
        s = substring(line, i, i+steps)
        mp=mp.union(s.boundary)
    xy_pixs = []
    for i in range(len(mp)):
        pX = mp[i].x
        pY = mp[i].y
        xy_pixs.append(rio.transform.rowcol(affine, pX,pY, math.floor))

    semi_distances = []
    for ii in range(len(mp)-1):
        pt1=Point(mp[ii].x,mp[ii].y)
        pt2=Point(mp[ii+1].x,mp[ii+1].y)    
        semi_distances.append(pt1.distance(pt2))  
        
    return(mp, xy_pixs, semi_distances, line.length)

def get_img_aff(file, dst_crs):
   
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    with rio.open(file) as src:
        if src.crs != dst_crs:
            transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
    
            kwargs = src.meta.copy()
            kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
            })
            savename=file.split('.')[0]+'_reprojected_'+file.split('.')[1]
            with rio.open(savename, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, 1),
                        destination=rio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic)
            dst_aff = transform
            dst_img = reshape_as_image(rio.open(savename).read())[:,:,0]
        else:
            dst_aff = src.transform
            dst_img = reshape_as_image(src.read())[:,:,0]
            savename = file
    return(savename, dst_aff, dst_img)
 
def profile_plotter(path, rows, cols,data_df,sub_ids, title,x_val, y_val, label_vals):
    count=1
    f, axs = plt.subplots(rows,cols,figsize=(15,15))
    for prof_id in sub_ids:
        id_label = set(data_df.loc[data_df['id']==prof_id,label_vals].to_list())
        plt.subplot(rows,cols,count)
        Title = title+' of ids '+str(sub_ids[0])+'-'+str(sub_ids[len(sub_ids)-1])
        plt.suptitle(Title, fontsize=16)
    
        for idl in id_label:
            
            mask = data_df[label_vals]==idl
            x = data_df.loc[mask,x_val]
            y=data_df.loc[mask,y_val]#.sort_values(ascending=False)
            plt.plot(x,y)#, label=idl)
            plt.title('Fault id: '+str(prof_id), fontsize=10)
    
            plt.yscale('linear')
            plt.xscale('linear')
            plt.ylabel(y_val+' (m)')
            plt.xlabel(x_val+' (m)')
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            plt.legend(sorted(set(id_label)),loc='best')
        count+=1
    plt.tight_layout()
    # plt.show()
    savename = os.path.dirname(path)+'/'+Title+'.png'
    plt.savefig(savename)
    plt.close()



def cut(line, distance):
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]
        
        
def cut_piece(line,distance, lgth):
    precut = cut(line,distance)[1]
    return precut    

def geom_densifier(src_geometry, num):
    # try:    
    

    if src_geometry.length <=2000:
        num*=4
    if src_geometry.length > 2000:
        num*=1.1
    lenSpace = np.linspace(0,src_geometry.length,int(src_geometry.length*num/100))
    tempList = []
    for space in lenSpace:
        points=src_geometry.interpolate(space)
        tempList.append([points.x,points.y])
   
    new_geom = LineString(tempList)
   
    return(new_geom)


def geodataframe_reproj(dst_folder, geodataframe_name, dst_crs):
    geodataframe = gpd.read_file(faults_linear_shapes).to_crs(dst_crs)
    geodataframe.columns = ['fault_id', 'geometry']
    geodataframe['Length(m)'] = geodataframe.apply(
        lambda row:row.geometry.length, axis=1)
    # geodataframe['Length(km)'] = geodataframe.apply(
    #     lambda row:row.geometry.length/1000, axis=1)
    new_name = dst_folder+'/'+os.path.basename(geodataframe_name).split('.')[0]+'_repr.shp'
    geodataframe.to_file(new_name, driver ='ESRI Shapefile')
    return(new_name, geodataframe)


def chunk_creator(iterable, JOBS):
    filerange = len(iterable)
    chunksize = round(filerange/JOBS)
    # import itertools
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            break
        yield chunk

def parallel_geodensifier(dst_folder, dataframe_name, ref, num):
    JOBS = 4
    dataframe = gpd.read_file(dataframe_name)
    chunks = []

    for c in chunk_creator(dataframe[ref].to_list(), JOBS):
        chunks.append(c)
    # Parallel processing
    chunk_results=[]
    results = []
    with tqdm(total=len(dataframe['geometry']),
             desc = 'Generating files',
             unit='File') as pbar:
        func = geom_densifier
        for i in range(len(chunks)):

            sub_chunk = chunks[i]
            
            files = [dataframe.iloc[ch-1].geometry for ch in sub_chunk]
            from joblib import Parallel, delayed
            results=(Parallel (n_jobs=JOBS)(delayed(func)(files[i],
                                                           num)
                                   for i in range(len(files))))
            pbar.update(len(results))
            [chunk_results.append(res) for res in results]
    for cr in range(len(chunk_results)):
        if ref == 'id':
            cr+=1
        mask = dataframe[ref] == cr
        dataframe.loc[mask, 'geometry'] = chunk_results[cr-1]
    print('Saving to shapefile')
    savename = dst_folder+'/'+os.path.basename(dataframe_name).split('.')[0]+'_densified_'+str(num)+'.shp'
    dataframe.to_file(savename, driver = 'ESRI Shapefile')
    return savename, dataframe

def FindMaxLength(lst):
    maxList = max((x) for x in lst)
    maxLength = max(len(x) for x in lst )
    for i in lst:
        if len(i) == maxLength:
            return i, maxLength

    return maxList, maxLength    

#####################################################  END BLOCK ZERO

################################# FIRST BLOCK 

root = Tk()
root.withdraw()
faults_linear_shapes = filedialog.askopenfilename(initialdir = "/",title = "Select file faults linear shapefile",filetypes = (("Esri Shapefile","*.shp"),("all files","*.*")))


dem_file = filedialog.askopenfilename(initialdir = "/",title = "Select DEM image",filetypes = (("Tiff/Tif",("*.tiff","*.tif")),("all files","*.*")))
slope_file = filedialog.askopenfilename(initialdir = "/",title = "Select SLOPE image",filetypes = (("Tiff/Tif",("*.tiff","*.tif")),("all files","*.*")))
aspect_file = filedialog.askopenfilename(initialdir = "/",title = "Select ASPECT image",filetypes = (("Tiff/Tif",("*.tiff","*.tif")),("all files","*.*")))
image = filedialog.askopenfilename(initialdir = "/",title = "Select basemap image",filetypes = (("Tiff/Tif",("*.tiff","*.tif")),("all files","*.*")))
basecrs = rio.open(image).crs

processing_folder  = make_folder(os.path.dirname(faults_linear_shapes),'Processing')
linear_reproj_gdf_file,linear_reproj_gdf = geodataframe_reproj(processing_folder, faults_linear_shapes, basecrs)
linear_densified_gdf_file, linear_densified_gdf=parallel_geodensifier(processing_folder, linear_reproj_gdf_file, 'fault_id', 0.1)


# print('Please create transcets with GIS software using densified shapefile and save in the processing folder. Then proceed')
# question('Proceed with processing?',['Y','y','n','N'])

################################# SECOND BLOCK 

transect_df_file = filedialog.askopenfilename(initialdir = "/",title = "Select file faults linear shapefile",filetypes = (("Esri Shapefile","*.shp"),("all files","*.*")))
status = None
while status != 'Pass':
    try:
        
        transect_df = gpd.read_file(transect_df_file)
        valid_col = transect_df['TR_ID']
        status = 'Pass'
    except Exception as e:
        print(e +' not found')
        print('\nPlease create the transect file in QGIS and try again\n')
        transect_df_file = filedialog.askopenfilename(initialdir = "/",title = "Select file faults linear shapefile",filetypes = (("Esri Shapefile","*.shp"),("all files","*.*")))
    
    

dem_new_file, dem_aff, dem_img = get_img_aff(dem_file, basecrs)
slope_new_file, slope_aff, slope_img = get_img_aff(slope_file, basecrs)
aspect_new_file, asp_aff, aspect_img = get_img_aff(aspect_file, basecrs)

###############

def transect_reshaper(dataframe_name):
    dataframe = gpd.read_file(transect_df_file)
    transect_stat_cols = ['fault_id','Fault L(m)','Tran_id','Tran L(m)','Tran AvL(m)','geometry']
    
    transect_gdf_upd = gpd.GeoDataFrame(columns=transect_stat_cols, dtype='int64')
    
    id_list = list(dataframe['fault_id'].unique())
    for idx in id_list:    
            mask = dataframe['fault_id']==(idx)
            masked_df = dataframe.loc[mask]
            sub_idxs = id_list = list(masked_df.TR_SEGMENT.unique())
            TranLen = []
            for sub_idx in sub_idxs:
                
                sub_mask = masked_df.TR_SEGMENT==sub_idx
                sub_df = masked_df.loc[sub_mask]
                df_id = sub_df.index.to_list()[0]
                # tran_displ = sub_df.displacement.max()
                geom = sub_df.geometry
                # if tran_displ >= np.max(tran_max_displs):
                # print('true')
                fault_len = masked_df.loc[df_id]['Length(m)']
                curr_len = geom.length[df_id]
                corr_len = geom.length[df_id]
                
                if  fault_len >50000:
                     corr_len = curr_len/1.1
                elif  45000 < fault_len <50000:
                       corr_len = curr_len/1.1
                elif  40000 < fault_len <45000:
                       corr_len = curr_len/1.15
                elif  35000 < fault_len <40000:
                       corr_len = curr_len/1.2
                elif  30000 < fault_len <35000:
                       corr_len = curr_len/1.3
                elif  25000 < fault_len <30000:
                       corr_len = curr_len/1.4
                elif  20000 < fault_len <25000:
                       corr_len = curr_len/1.6
                elif  15000 < fault_len <20000:
                       corr_len = curr_len/1.65
                elif  10000 < fault_len <15000:
                       corr_len = curr_len/2.1
                elif  5000 < fault_len <10000:
                       corr_len = curr_len/2.8
                else:
                    corr_len = curr_len/3
              
                if idx in [47]:
                    corr_len = curr_len/5
                if idx in [44]:
                    corr_len = curr_len/1.2
                    if 50 > sub_idx >=25:
                        corr_len = curr_len/3
                    if sub_idx <=5:
                        corr_len = curr_len*1.2
                if idx in [43]:
                    corr_len = curr_len/1.15
                    if 52 >= sub_idx >=50:
                        corr_len = curr_len/2.5
                    if 42 >= sub_idx >=37:
                        corr_len = curr_len/2
                    if 24 >= sub_idx >=13:
                        corr_len = curr_len/2
                    if sub_idx <=3:
                        corr_len = curr_len/2.5
                if idx in [40]:
                    corr_len = curr_len/1.1
                    if sub_idx >=36:
                        corr_len = curr_len/4
                if sub_idx <=6:
                        corr_len = curr_len/3
                if idx in [39]:
                    corr_len = curr_len/1
                    if sub_idx <=3:
                        corr_len = curr_len/4
                    if sub_idx>=52:
                        corr_len = curr_len/2
                if idx in [37, 38]:
                    corr_len = curr_len/1.1
                    if sub_idx >=17:
                        corr_len = curr_len/2
                if idx in [35,36]:
                    corr_len = curr_len/1.4
                if idx in [29,30]:
                    corr_len = curr_len/1.1
                if idx in [28]:
                    corr_len = curr_len/2
                    if sub_idx <=6:
                        corr_len = curr_len/1.5
                if idx in [24]:
                    corr_len = curr_len/2.5
                    if 17 > sub_idx >= 14:
                        corr_len = curr_len/1.1
                    else: corr_len = curr_len/4
                if idx in [23]:
                    corr_len = curr_len/2.5
                    if sub_idx >= 10:
                        corr_len = curr_len/5
                if idx in [22]:
                    corr_len = curr_len/2
                    if sub_idx < 10:
                        corr_len = curr_len/5
                if idx in [21]:
                    corr_len = curr_len/2
                    if sub_idx >= 10:
                        corr_len = curr_len/5
                if idx in [20]:
                    if sub_idx <= 3:
                        corr_len = curr_len/2.5
                    if sub_idx >= 20:
                        corr_len = curr_len/2.5
                if idx in [19]:
                    if sub_idx <= 1:
                        corr_len = curr_len/2.5
                    if sub_idx >= 11:
                        corr_len = curr_len/2.5
                if idx in [16]:
                    corr_len = curr_len/1.5
                    if 42 >= sub_idx >= 30:
                        corr_len = curr_len/2.5
                    # else: corr_len = curr_len/2
                if idx in [15]:                    
                    if sub_idx <=7:
                        corr_len = curr_len/1.5
                if idx in [14, 13]:
                    corr_len = curr_len/1.6
                if idx in [8]:
                    if sub_idx >6:
                        corr_len = curr_len/3
                    else:
                        corr_len = curr_len/1.3    
                if idx in [7]:
                    if sub_idx <12:
                        corr_len = curr_len/4
                    else:
                        corr_len = curr_len/1.4
                if idx in [3]:
                    corr_len = curr_len/1.1
                if idx in [2]:
                    corr_len = curr_len/1.1
                    if sub_idx >=48:
                        corr_len = curr_len/2.5  
                if idx in [1]:
                    corr_len = curr_len/1.1
                    if sub_idx <=10:
                        corr_len = curr_len/2.5
               
                geom = cut(geom[df_id],corr_len)[0]
                
                start = 0
                stop = math.ceil(geom.length)#.length[df_id])
                fault_length = sub_df.loc[df_id]['Length(m)']
               
                    
                steps = len(sub_df)
                space = np.linspace(start, stop, 100)
                mp = shapely.geometry.MultiPoint()
    
                for i in space:
                    
                    s = substring(geom, i, i+steps)
                    mp=mp.union(s.boundary)
                xy_pixs = []
    
                for i in mp:
                    pX = i.x
                    pY = i.y
                    xy_pixs.append(rio.transform.rowcol(dem_aff, pX,pY))
               
                height_profile =[]
                slope_profile = []
                aspect_profile = []
                for ii in xy_pixs:
                    height_profile.append(dem_img[ii])
                    slope_profile.append(slope_img[ii])
                    aspect_profile.append(aspect_img[ii])
                
                min_indxs = np.where(height_profile == np.amin(height_profile))
                min_indx =min_indxs[len(min_indxs)//2][0]
                min_val = height_profile[min_indx]
                min_profs = list(split_after(height_profile,lambda x: x==min_val, maxsplit=2))
                min_prof, minProfLength = FindMaxLength(min_profs)
                
                x = np.linspace(0, stop, num=len(height_profile), endpoint=True)[0:minProfLength]
                
                #update_geometry
                new_length = math.ceil(x[-1])
                max_tra_len = int(fault_length*1)
                if  new_length >=max_tra_len:
                    new_length = max_tra_len
                TranLen.append(new_length)
                
                new_geom = cut(geom,new_length)[0]
                
                temp_data = {'fault_id':idx,
                 'Fault L(m)':int(sub_df['Length(m)']),
                 'Tran_id':sub_idx,
                 'Tran L(m)':new_length,
                 'geometry':new_geom
                 }
                
                temp_gdf = gpd.GeoDataFrame.from_records([temp_data])
                transect_gdf_upd = pd.concat([transect_gdf_upd,temp_gdf], ignore_index=True)#,join='inner')
            
            transect_gdf_upd.loc[mask, 'Tran AvL(m)'] = np.mean(TranLen)
            
    transect_gdf_upd['Length(m)']=transect_gdf_upd.geometry.length
    savename = processing_folder+'/'+os.path.basename(transect_df_file).split('.')[0]+'_reshaped'+'.shp'
    
    transect_gdf_upd.crs=basecrs
    transect_gdf_upd.to_file(savename, driver = 'ESRI Shapefile')       
    return(savename, transect_gdf_upd)      

##################

transect_gdf_reshaped_file, transect_gdf_reshaped =  transect_reshaper(transect_df_file)

################# USE PREVIOUS FILE WITH QPROF AND READ THE CREATED FILE


################################# THIRD BLOCK 

transect_reshaped_profiles_shp = filedialog.askopenfilename(initialdir = "/",title = "Select file transect reshaped profiles shapefile",filetypes = (("Esri Shapefile","*.shp"),("all files","*.*")))
transect_reshaped_profiles_df = gpd.read_file(transect_reshaped_profiles_shp)

def transect_qprof_unifier(dst_folder, dataframe_name):
    
    transect_profiles_shp_df = gpd.read_file(dataframe_name)
    transect_profiles_shp_df.columns=['fault_id','tran_id','sub_id','cds2d','height','displacement','dir_slope','geometry']
    id_list = list(transect_profiles_shp_df.fault_id.unique())
    transect_stat_cols = ['fault_id','tran_id','Heave(m)','Throw(m)','Disp(m)','geometry']
    transect_stat_df = gpd.GeoDataFrame(columns=transect_stat_cols, dtype='int64')
    for idx in id_list:    
        mask = transect_profiles_shp_df['fault_id']==(idx)
        
        # masked_df=transect_profiles_shp_df.loc[mask]
        masked_df = transect_profiles_shp_df.loc[mask]
        
        # multiLine = ops.linemerge(geometry.MultiLineString(masked_df['geometry'].to_list()))
        # temp_df = pd.dataframe()
        # temp_df['fault_id']=idx
        # temp_df['fault_id']=idx
        
        sub_idxs = id_list = list(masked_df.tran_id.unique())
        for sub_idx in sub_idxs:
            # print(idx)
            sub_mask = masked_df['tran_id']==sub_idx
            geoms = masked_df.loc[sub_mask,'geometry'].to_list()
            multiline = ops.linemerge(geometry.MultiLineString(geoms))
            data={'fault_id':int(idx),
                  'tran_id':int(sub_idx),
                  'Heave(m)':multiline.length,
                  'Throw(m)':int(max(masked_df.loc[sub_mask,'height'].to_list())-min(masked_df.loc[sub_mask,'height'].to_list())),
                  'Disp(m)':max(masked_df.loc[sub_mask,'displacement'].to_list()),
                  'geometry':[multiline]}
            
            temp_df = gpd.GeoDataFrame(data,index=[0])
            transect_stat_df= pd.concat([transect_stat_df,temp_df],ignore_index=True)
    savename = dst_folder+'/'+os.path.basename(dataframe_name).split('.')[0]+'_filtered'+'.shp'
    transect_stat_df.to_file(savename, driver = 'ESRI Shapefile')
    return savename, transect_stat_df
        
transect_reshaped_profiles_df_filtered_file ,transect_reshaped_profiles_df_filtered = transect_qprof_unifier(processing_folder, transect_reshaped_profiles_shp)
   
transect_reshaped_profiles_df_filtered_den_file, transect_reshaped_profiles_df_filtered_den = parallel_geodensifier(processing_folder,
                                                                          transect_reshaped_profiles_df_filtered_file,
                                                                          'tran_id',
                                                                          0.2)



def transect_profile_stats(dst_folder, dataframe_name, linear_df):
    stat_df = gpd.read_file(dataframe_name)
    new_stat_df = gpd.GeoDataFrame(dtype='float32')
    fault_ids = list(stat_df.fault_id.unique())
    for fidx in fault_ids:
        fault_mask = stat_df['fault_id'] == fidx
               
        disp_arr = stat_df.loc[fault_mask]['Disp(m)']
        min_disp = min(disp_arr)
        max_disp = max(disp_arr)
        mean_disp = np.mean(disp_arr)
        min_disp_ids = list( np.where(disp_arr == np.amin(disp_arr))[0])
        min_disp_id = [el+1 for el in min_disp_ids]
        max_disp_ids = list(np.where(disp_arr == np.amax(disp_arr))[0])
        max_disp_id = [el+1 for el in max_disp_ids]
        
        heave_arr = stat_df.loc[fault_mask]['Heave(m)']
        min_heave = min(heave_arr)
        max_heave = max(heave_arr)
        mean_heave = np.mean(heave_arr)
        min_heave_ids = list(np.where(heave_arr == np.amin(heave_arr))[0])
        min_heave_id = [el+1 for el in min_heave_ids][0]
        max_heave_ids = list(np.where(heave_arr == np.amax(heave_arr))[0])
        max_heave_id = [el+1 for el in max_heave_ids][0]
        
        throw_arr = stat_df.loc[fault_mask]['Throw(m)']
        min_throw = min(throw_arr)
        max_throw = max(throw_arr)
        mean_throw = np.mean(throw_arr)
        min_throw_ids = list(np.where(throw_arr == np.amin(throw_arr))[0])
        min_throw_id = [el+1 for el in min_throw_ids][0]
        max_throw_ids = list(np.where(throw_arr == np.amax(throw_arr))[0])
        max_throw_id = [el+1 for el in max_throw_ids][0]
        fault_stats = {
        'Fault id':fidx,
        'MinDsp(m)':min_disp,
        'MinDspID':min_disp_id,
        'MaxDsp(m)':max_disp,
        'MaxDspID':max_disp_id,
        'MeanDisp(m)':mean_disp,
        'MinHv(m)':min_heave,
        'MinHvID':min_heave_id,
        'MaxHv(m)':max_heave,
        'MaxHvID':max_heave_id,
        'MeanHv(m)':mean_heave,
        'MinTw(m)':min_throw,
        'MinTrwID':min_throw_id,
        'MaxTrw(m)':max_throw,
        'MaxTrwID':max_throw_id,
        'MeanTrw(m)':mean_throw        
        }
        fault_stat_df = gpd.GeoDataFrame(fault_stats)
        fault_stat_df['geometry']=linear_df.loc[fidx-1].geometry
        fault_stat_df['Length(m)']=linear_df.loc[fidx-1].geometry.length
        new_stat_df = pd.concat([new_stat_df,fault_stat_df])
        new_stat_df.crs = stat_df.crs
    savename = dst_folder+'/'+os.path.basename(dataframe_name).split('.')[0]+'_statistics'+'.shp'
    new_stat_df.to_file(savename, driver = 'ESRI Shapefile')
    return(savename, new_stat_df)

statistic_df_file, statistic_df = transect_profile_stats(processing_folder,
                                      transect_reshaped_profiles_df_filtered_den_file,
                                      linear_densified_gdf)


def transect_profile_creator(stat_dataframe, ref, linspace_num):
    profiles_geodataframe = pd.DataFrame(dtype='float32')
    len_geom = len(list(stat_dataframe[ref]))
    with tqdm(total=len_geom,
             desc = 'Creating transect profiles',
             unit='File') as pbar:
        
        for indx in range(len_geom):
            xy, xy_pixs, semi_dist, length = get_MP_PX_coords(indx, stat_dataframe, dem_aff, linspace_num)
            height_profile =[]
            slope_profile = []
            aspect_profile = []
            for ii in xy_pixs:
                height_profile.append(dem_img[ii])
                slope_profile.append(slope_img[ii])
                aspect_profile.append(aspect_img[ii])
            if ref == 'tran_id':
                tran_id = int(stat_dataframe.iloc[indx]['tran_id'])
                fault_id = int(stat_dataframe.iloc[indx]['fault_id'])
                line_df = pd.DataFrame(columns=['fault_id','tran_id',
                                                'Distance',#'Heave(m)',
                                                'Slope','Aspect'])
                min_indxs = np.where(height_profile == np.amin(height_profile))[0]
                min_indx =min_indxs[len(min_indxs)//2]
                min_prof = height_profile[0:min_indx]
                line_df['Height']=min_prof            
                line_df['Slope']=slope_profile[0:min_indx]
                line_df['Aspect']=aspect_profile[0:min_indx]
                start = 0
                stop = math.ceil(length)
                if min_prof == []:
                    min_prof = height_profile
                line_df['Distance']=np.linspace(start, stop, len(min_prof))[0:min_indx]
                line_df['tran_id']=int(tran_id)
                line_df['fault_id']=int(fault_id)
                
            else:
                fault_id = int(stat_dataframe.iloc[indx]['fault_id'])
                tran_id = None
                line_df = pd.DataFrame(columns=['fault_id',
                                               'Distance',#'Heave(m)',
                                               'Slope','Aspect'])

                line_df['Height']=height_profile   
                line_df['Slope']=slope_profile
                line_df['Aspect']=aspect_profile
                start = 0
                stop = math.ceil(length)
                line_df['Distance']=np.linspace(start, stop, len(height_profile))
                line_df['fault_id']=int(fault_id)
            profiles_geodataframe=pd.concat([profiles_geodataframe,line_df], ignore_index=True)
            pbar.update(1)
        return profiles_geodataframe
    

def geodf_combiner(dst_folder, statistic_dataframe, transect_profile_df, name):
    fault_ids = statistic_df['Fault id'].to_list()
    linear_profiles_calcs = transect_profile_creator(transect_profile_df,'fault_id', 100)
    for idxx in fault_ids:
        min_aspect = min(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Aspect'])
        max_aspect = max(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Aspect'])
        mean_aspect = np.mean(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Aspect'])
        stdev_aspect = np.std(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Aspect'])
        min_slope = min(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Slope'])
        max_slope = max(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Slope'])
        mean_slope = np.mean(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Slope'])
        stdev_slope = np.std(linear_profiles_calcs.loc[linear_profiles_calcs['fault_id']==idxx, 'Slope'])
        maxDLratio = statistic_df.loc[statistic_df['Fault id']==idxx, 'MaxDsp(m)']/statistic_df.loc[statistic_df['Fault id']==idxx, 'Length(m)']
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MinAspect'] = min_aspect
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MaxAspect'] = max_aspect
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MeanAspect'] = mean_aspect
        statistic_df.loc[statistic_df['Fault id']==idxx, 'StDevAspect'] = stdev_aspect
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MinSlope'] = min_slope
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MaxSlope'] = max_slope
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MeanSlope'] = mean_slope
        statistic_df.loc[statistic_df['Fault id']==idxx, 'StDevSlope'] = stdev_slope
        statistic_df.loc[statistic_df['Fault id']==idxx, 'MaxD/L'] = maxDLratio
    savename = dst_folder+'/'+os.path.basename(name).split('.')[0]+'_complete_statistics'+'.shp'
    statistic_df.to_file(savename, driver = 'ESRI Shapefile')
    return(savename, statistic_df)


statistic_df_file, statistic_df = geodf_combiner(processing_folder, statistic_df, linear_densified_gdf, linear_densified_gdf_file) 

def aperture_calc(dst_folder, statistic_dataframe_file):
    stat_df = gpd.read_file(statistic_df_file)
    fault_ids = statistic_df['Fault id'].to_list()
    

    pairs = []
    pairs_ids = np.arange(fault_ids[0], fault_ids[-1], 2)
    
    for ii in pairs_ids:
        pairs.append((fault_ids[ii-1],fault_ids[ii]))

    
    stat_df['MeanAprt(m)'] = np.nan
    for pair in pairs:
        mean_semi_aperture_1 = stat_df.loc[stat_df['Fault id'] == pair[0]]['MeanHv(m)'].values
        mean_semi_aperture_2 = stat_df.loc[stat_df['Fault id'] == pair[1]]['MeanHv(m)'].values
        mean_aperture = mean_semi_aperture_1+mean_semi_aperture_2
        stat_df.loc[stat_df['Fault id']==pair[0], 'MeanAprt(m)'] = mean_aperture
        stat_df.loc[stat_df['Fault id']==pair[1], 'MeanAprt(m)'] = mean_aperture
        
        max_semi_aperture_1 = stat_df.loc[stat_df['Fault id'] == pair[0]]['MaxHv(m)'].values
        max_semi_aperture_2 = stat_df.loc[stat_df['Fault id'] == pair[1]]['MaxHv(m)'].values
        max_aperture = max_semi_aperture_1+max_semi_aperture_2
        stat_df.loc[stat_df['Fault id']==pair[0], 'MaxAprt(m)'] = max_aperture
        stat_df.loc[stat_df['Fault id']==pair[1], 'MaxAprt(m)'] = max_aperture
        
        min_semi_aperture_1 = stat_df.loc[stat_df['Fault id'] == pair[0]]['MinHv(m)'].values
        min_semi_aperture_2 = stat_df.loc[stat_df['Fault id'] == pair[1]]['MinHv(m)'].values
        min_aperture = min_semi_aperture_1+min_semi_aperture_2
        stat_df.loc[stat_df['Fault id']==pair[0], 'MinAprt(m)'] = min_aperture
        stat_df.loc[stat_df['Fault id']==pair[1], 'MinAprt(m)'] = min_aperture
    savename = dst_folder+'/'+os.path.basename(statistic_dataframe_file).split('.')[0]+'.shp'
    stat_df.to_file(savename, driver = 'ESRI Shapefile')
    return(savename, stat_df)

statistic_df_file, statistic_df = aperture_calc(processing_folder, statistic_df_file)

################################# END THIRD BLOCK 


################################# FOURTH BLOCK
#statistic_df = gpd.read_file('/media/hyradus/I-DATS/Working/Erica-FFC/Komarov/Processing_original/faults-Komarovv03_repr_densified_0_complete_statistics.shp')

######## PLOT maxD/Length
from scipy.optimize import curve_fit
# from numpy.polynomial.polynomial import polyfit
# plt.scatter(x, y,marker='.', label='Data for regression')


x = statistic_df['Length(m)']
y = statistic_df['MaxDsp(m)']
s = statistic_df['MeanAprt(m)']
#xerr = statistic_df['StDevSlope']
yerr = statistic_df['StDevSlope']*10
#c = statistic_df['Length(m)']
c = statistic_df['MeanAspect']
def powlaw(x, a, b) :
    return a * np.power(x, b)
def linlaw(x, a, b) :
    return a + x * b

def curve_fit_log(xdata, ydata) :
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)

popt_log, pcov_log, ydatafit_log = curve_fit_log(x,y)
fig, ax = plt.subplots()
ax.scatter(x, y, s=50, cmap='rainbow')#, c = np.random.rand(50))
# xerr = np.std(x)
# yerr = np.std(y)
ax.errorbar(x, y, xerr=None, yerr=yerr, fmt='',linestyle='')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Length (m)')
ax.set_ylabel('Max Displacement (m)')
# plt.scatter(x, y,marker='.', label='Data for regression')
popt, pcov = curve_fit(lambda fx,a,b: a*fx**-b,  x,  y)

power_y = popt_log[0]*x**popt_log[1]
ax.plot(sorted(x), sorted(ydatafit_log))


correlation_matrix = np.corrcoef(x, ydatafit_log)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Length (m)')
ax.set_ylabel('Max Displacement (m)')
ax.set_title("y=$%.6fx^{%.6f}$ \n$R^{2}$: %.6f"%(popt_log[0],popt_log[1],r_squared))
fname = processing_folder+'/Komarov_MaxDisp_Length_normalized.eps'
plt.savefig(fname, format='eps')

plt.savefig(fname.split('.eps')[0]+'.png', format='png')
# plt.title("y=%.6fx^%.6f - R^2"%(popt[0],popt[1]))1
######## END PLOT maxD/Length





######## PLOT maxd/length ratio per fault

x = statistic_df['Fault id']
y = statistic_df['MaxD/L']

fig, ax = plt.subplots()
ax.scatter(x, y, s=50, cmap='rainbow')#, c = np.random.rand(50))
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel('Fault ID')
ax.set_ylabel('MaxD/L ratio')
# ax.set_xticks(np.arange(0,50, step=1))
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
# ax.set_title("y=$%.6fx^{%.6f}$ \n$R^{2}$: %.6f"%(popt_log[0],popt_log[1],r_squared))
ax.set_title('Komarov MaxD/L')
fname = processing_folder+'/Komarov_maxDLength_ratio.eps'
plt.savefig(fname, format='eps')

plt.savefig(fname.split('.eps')[0]+'.png', format='png')

######## PLOT max aperture /max displ

x = statistic_df['MaxAprt(m)']
y = statistic_df['MaxDsp(m)']
fig, ax = plt.subplots()
ax.scatter(x, y, s=50, cmap='rainbow')#, c = np.random.rand(50))
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Max Aperture (m)')
ax.set_ylabel('Max Displacement (m)')
# ax.set_xticks(np.arange(0,50, step=1))
# ax.set_title("y=$%.6fx^{%.6f}$ \n$R^{2}$: %.6f"%(popt_log[0],popt_log[1],r_squared))
ax.set_title('Komarov Max Displacement/Max Aperture')
fname = processing_folder+'/Komarov_maxD-maxA log.eps'
plt.savefig(fname, format='eps')

plt.savefig(fname.split('.eps')[0]+'.png', format='png')

######## PLOT max aperture /max displ

x = statistic_df['Length(m)']
y = statistic_df['MaxAprt(m)']
fig, ax = plt.subplots()
ax.scatter(x, y, s=50, cmap='rainbow')#, c = np.random.rand(50))
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Length (m)')
ax.set_ylabel('Max Aperture (m)')
# ax.set_xticks(np.arange(0,50, step=1))
# ax.set_title("y=$%.6fx^{%.6f}$ \n$R^{2}$: %.6f"%(popt_log[0],popt_log[1],r_squared))
ax.set_title('Komarov Length/Aperture')
fname = processing_folder+'/Komarov_Ape-Len log.eps'
plt.savefig(fname, format='eps')
plt.savefig(fname.split('.eps')[0]+'.png', format='png')



def transect_profile_plotter(Transect_profiles_calcs, fault_number, density):
    temp_plot_mask = Transect_profiles_calcs.fault_id==fault_number
    temp_plot_df = Transect_profiles_calcs.loc[temp_plot_mask]
    
    temp_trans_ids = temp_plot_df.tran_id.unique()
    start = int(temp_trans_ids[0])
    stop = int(temp_trans_ids[-1])
    steps = density
    sub_samp = np.arange(start, stop, steps)
    # temp_fault_ids = temp_plot_df.fault_id.unique()
    for temp_idx in sub_samp:
        # print(temp_idx)
        temp_sub_mask = temp_plot_df.tran_id == temp_idx
        temp_sub_df = temp_plot_df.loc[temp_sub_mask]
        x = temp_sub_df['Distance']#[min_indx:len(height_profile)]
        y = temp_sub_df['Height']#[min_indx:len(height_profile)]

        plt.legend(sub_samp)
        plt.plot(x, y)

    plt.xlabel('Length (m)')
    plt.ylabel('Elevation (m)')
    plt.title("Sample of transect topographic profiles for fault: %.i" %fault_number)
    plt.grid(True)
    plt.show()










# ######## PLOT maxD/Length
# from scipy.optimize import curve_fit

# x = statistic_df['Length(m)']
# y = statistic_df['MaxDsp(m)']
# s = statistic_df['MeanAprt(m)']
# yerr = statistic_df['StDevSlope']*10
# c = statistic_df['MeanAspect']

# def powlaw(x, a, b) :
#     return a * np.power(x, b)
# def linlaw(x, a, b) :
#     return a + x * b

# def curve_fit_log(xdata, ydata) :
#     """Fit data to a power law with weights according to a log scale"""
#     # Weights according to a log scale
#     # Apply fscalex
#     xdata_log = np.log10(xdata)
#     # Apply fscaley
#     ydata_log = np.log10(ydata)
#     # Fit linear
#     popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
#     #print(popt_log, pcov_log)
#     # Apply fscaley^-1 to fitted data
#     ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
#     # There is no need to apply fscalex^-1 as original data is already available
#     return (popt_log, pcov_log, ydatafit_log)

# popt_log, pcov_log, ydatafit_log = curve_fit_log(x,y)
# fig, ax = plt.subplots()
# ax.scatter(x, y, s=50, cmap='rainbow')#, c = np.random.rand(50))
# ax.errorbar(x, y, xerr=None, yerr=yerr, fmt='',linestyle='')
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel('Length (m)')
# ax.set_ylabel('Max Displacement (m)')
# popt, pcov = curve_fit(lambda fx,a,b: a*fx**-b,  x,  y)

# power_y = popt_log[0]*x**popt_log[1]
# ax.plot(sorted(x), sorted(ydatafit_log))


# correlation_matrix = np.corrcoef(x, ydatafit_log)
# correlation_xy = correlation_matrix[0,1]
# r_squared = correlation_xy**2
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel('Length (m)')
# ax.set_ylabel('Max Displacement (m)')
# ax.set_title("y=$%.6fx^{%.6f}$ \n$R^{2}$: %.6f"%(popt_log[0],popt_log[1],r_squared))
# fname = processing_folder+'/Komarov_MaxDisp_Length_normalized.eps'
# plt.savefig(fname, format='eps')

# plt.savefig(fname.split('.eps')[0]+'.png', format='png')


fault_ids = linear_densified_gdf['fault_id'].to_list()
transect_reshaped_profiles_df_filtered_den['Length(m)']=np.nan
for idxx in fault_ids:
    mask = linear_densified_gdf['fault_id']==idxx
    length = list(linear_densified_gdf.loc[mask, 'Length(m)'])[0]
    temp_mask = transect_reshaped_profiles_df_filtered_den['fault_id']==idxx
    transect_reshaped_profiles_df_filtered_den.loc[temp_mask,'Length(m)'] = length

from sklearn.preprocessing import scale, normalize, MinMaxScaler
x_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaler = MinMaxScaler(feature_range=(0,1))




fault_ids = list(transect_reshaped_profiles_df_filtered_den['fault_id'].unique())#.to_list()

plot_chunks = []
  
for c in chunk_creator(fault_ids, 5):
      plot_chunks.append(c)
plt_chnk = plot_chunks[0]      
cols = 2


for plt_chnk in plot_chunks:
    fig, axs = plt.subplots(int(len(plt_chnk)/cols), cols, figsize=(10,10))    
    axs = axs.ravel()
    
    for i in range(len(plt_chnk)):
        
        idxx=plt_chnk[i]
        fault_length = transect_reshaped_profiles_df_filtered_den.loc[idxx]['Length(m)']
        temp_mask = transect_reshaped_profiles_df_filtered_den['fault_id']==idxx
        tran_num = max(transect_reshaped_profiles_df_filtered_den.loc[temp_mask, 'tran_id'].to_list())
        tran_disp = transect_reshaped_profiles_df_filtered_den.loc[temp_mask, 'Disp(m)'].to_list()
        rat = (np.mean(transect_reshaped_profiles_df_filtered_den.loc[temp_mask, 'Throw(m)'].to_list()))/np.mean(transect_reshaped_profiles_df_filtered_den.loc[temp_mask, 'Heave(m)'].to_list())
            
        xs = np.linspace(0, fault_length, tran_num)
        
        y = tran_disp
        
        xs_norm = x_scaler.fit_transform(xs.reshape(-1,1))
        y_norm = y_scaler.fit_transform(np.array(y).reshape(-1,1))
        
        axs[i].plot(xs_norm,y_norm)
        
        axs[i].grid(True)
        # from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        
        axs[i].set_title('Displacement distribution along fault n°: %.i'%idxx)
        axs[i].set_xlabel( 'Length')
        
        axs[i].set_ylabel('Max Displacement')
        axs[i].set_xticks((-1,0,1))
        xlabels = ['-L/2','0','L/2']
        axs[i].set_xticklabels(xlabels)
        
    plt.tight_layout()
    plt.show()
    id1 = str(plt_chnk[0])
    id2=str(plt_chnk[-1])
    fname = processing_folder+'/Komarov_Displacement_along_faults_'+"{0}-{1}".format(id1, id2)+'normalized.png'
    
    plt.savefig(fname, dpi=150)
    fname = processing_folder+'/Komarov_Displacement_along_faults_'+"{0}-{1}".format(id1, id2)+'normalized.pdf'
    plt.savefig(fname, format='pdf')


################################# END FOURTH BLOCK 