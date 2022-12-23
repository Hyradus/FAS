import os
import shutil
from datetime import datetime
import itertools
from more_itertools import split_after
import numpy as np
import numpy.ma as ma
import pandas as pd
import joblib
from joblib import Parallel, delayed
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
from pyproj import Geod
geod = Geod('+a=1737400.0')
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
    #for i in range(len(mp)):
    for i in mp.geoms:
        #pX = mp[i].x
        #pY = mp[i].y
        pX = i.x        
        pY = i.y
        xy_pixs.append(rio.transform.rowcol(affine, pX,pY, math.floor))

    semi_distances = []
    for ii in range(len(mp.geoms)-1):
        pt1=Point(mp.geoms[ii].x,mp.geoms[ii].y)
        pt2=Point(mp.geoms[ii+1].x,mp.geoms[ii+1].y)    
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
            savename=file.split('.shp')[0]+'_reprojected_'+file.split('.shp')[1]
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

def geom_densifier(src_geometry, num, dem_res):
    # try:        
    samples = int(src_geometry.length/dem_res/num)
    lenSpace = np.linspace(0,src_geometry.length,samples)

    tempList = []
    for space in lenSpace:
        points=src_geometry.interpolate(space)
        tempList.append([points.x,points.y])
   
    new_geom = LineString(tempList)
   
    return(new_geom)
import itertools
from more_itertools import split_after
from shapely.geometry import Point, LineString


def geodataframe_reproj(dst_folder, geodataframe_name, dst_crs):
    geodataframe = gpd.read_file(geodataframe_name)#.to_crs(dst_crs)
    geodataframe.columns = ['fault_id', 'geometry']
    
    # geodataframe['Length(km)'] = geodataframe.apply(
    #     lambda row:row.geometry.length/1000, axis=1)
    new_name = dst_folder+'/'+os.path.basename(geodataframe_name).split('.shp')[0]+'_repr.gpkg'
    geodataframe.to_file(new_name, driver ='GPKG')
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



def FindMaxLength(lst):
    maxList = max((x) for x in lst)
    maxLength = max(len(x) for x in lst )
    for i in lst:
        if len(i) == maxLength:
            return i, maxLength

    return maxList, maxLength    

def FindMinLength(lst):
    minList = min((x) for x in lst)
    minLength = min(len(x) for x in lst )
    for i in lst:
        if len(i) == minLength:
            return i, minLength

    return minList, minLength    

def dem_profiler(xarr,line, n_samples):
    profile = []
    points = []

    for i in range(n_samples):
        # get next point on the line
        point = line.interpolate(i / n_samples - 1., normalized=True)
        # access the nearest pixel in the xarray
        value = xarr.sel(x=point.x, y=point.y, method="nearest").data
        profile.append(value)
        points.append(point)
        
    return(np.where(np.isnan(profile), ma.array(profile, mask=np.isnan(profile)).mean(axis=0), profile), points)



def parallel_geodensifier(dst_folder, dataframe_name, ref, num, dem_res):
    JOBS = joblib.cpu_count()
    dataframe = gpd.read_file(dataframe_name)
    #new_dataframe = gpd.GeoDataFrame(columns=dataframe.columns)
    new_dataframe = dataframe[list(dataframe.columns)].copy()
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
            
            results=(Parallel (n_jobs=JOBS)(delayed(func)(files[i],
                                                           num, dem_res)
                                   for i in range(len(files))))
            pbar.update(len(results))
            for res in results:
                chunk_results.append(res)
    new_dataframe['geometry']=chunk_results
    print('Saving to shapefile')
    savename = dst_folder+'/'+os.path.basename(dataframe_name).split('.gpkg')[0]+'_densified_'+str(num)+'.gpkg'
    new_dataframe.to_file(savename, driver = 'GPKG')
    return savename, new_dataframe, chunk_results

def giveline(linestring, distance):
    "From a input shapely linestring, return a perpendicular linestring with the length equal to intput distance"
    pline = linestring.parallel_offset(distance=distance)
    return LineString([linestring.centroid, pline.centroid])


def transecter(linear_densified_gdf,geographic_linear_densified_gdf, l, length_param):#,geoLengths, segments,temp_datas):
    geoLengths = []
    segments = []
    temp_datas= []
    
    if l != len(geographic_linear_densified_gdf)-1:
        points = list(zip([x for x in linear_densified_gdf.iloc[l].geometry.xy[0]],[y for y in linear_densified_gdf.iloc[l].geometry.xy[1]]))
        geo_points = list(zip([x for x in geographic_linear_densified_gdf.iloc[l].geometry.xy[0]],[y for y in geographic_linear_densified_gdf.iloc[l].geometry.xy[1]]))
        geodetic_length = 0
        max_length = geographic_linear_densified_gdf.iloc[l][length_param]

        for i in range(len(points)):
            if i != len(points)-1:
                geo_line_string = LineString([Point(geo_points[i]),Point(geo_points[i+1])])
                line_string = LineString([Point(points[i]),Point(points[i+1])])
                geodetic_length+=geod.geometry_length(geo_line_string)            
                segments.append(line_string)
            else:
                geo_line_string = LineString([Point(geo_points[-2]),Point(geo_points[-1])])
                line_string = LineString([Point(points[-2]),Point(points[-1])])
                geodetic_length+=geod.geometry_length(geo_line_string)
                segments.append(line_string)
            transect_geom=giveline(line_string,-max_length)
            temp_data = {'Fault ID':linear_densified_gdf.iloc[l]['fault_id'],
                     'Segment L(m)':geodetic_length,                 
                     'geometry':transect_geom
                     }
            temp_datas.append(temp_data)
            #transect_gdf_tmp=gpd.GeoDataFrame([temp_data],crs=basecrs)
    else:
        points = list(zip([x for x in linear_densified_gdf.iloc[-2].geometry.xy[0]],[y for y in linear_densified_gdf.iloc[-1].geometry.xy[1]]))
        geo_points = list(zip([x for x in geographic_linear_densified_gdf.iloc[-2].geometry.xy[0]],[y for y in geographic_linear_densified_gdf.iloc[-1].geometry.xy[1]]))
        geodetic_length = 0
        max_length = geographic_linear_densified_gdf.iloc[l][length_param]
        for i in range(len(points)):
            if i != len(points)-1:
                geo_line_string = LineString([Point(geo_points[i]),Point(geo_points[i+1])])
                line_string = LineString([Point(points[i]),Point(points[i+1])])
                geodetic_length+=geod.geometry_length(geo_line_string)            
                segments.append(line_string)
            else:
                geo_line_string = LineString([Point(geo_points[-2]),Point(geo_points[-1])])
                line_string = LineString([Point(points[-2]),Point(points[-1])])
                geodetic_length+=geod.geometry_length(geo_line_string)
                segments.append(line_string)
            transect_geom=giveline(line_string,-max_length)
            temp_data = {'Fault ID':linear_densified_gdf.iloc[l]['fault_id'],
                     'Segment L(m)':geodetic_length,                 
                     'geometry':transect_geom
                     }
            temp_datas.append(temp_data)
    geoLengths.append(geodetic_length)
            #transect_gdf_tmp=gpd.GeoDataFrame([temp_data],crs=basecrs)
    return(geoLengths, segments,temp_datas)


def parallel_transectifier(linear_densified_gdf, length_param):
    from pyproj import Geod
    geod = Geod('+a=1737400.0')
    geographic_linear_densified_gdf=linear_densified_gdf.to_crs(GCS_Moon_2000)
    JOBS = joblib.cpu_count()

    results = []
    geoLengths = []
    segments = []
    temp_datas= []
    chunks = []
    chunk_results=[]

    for c in chunk_creator(geographic_linear_densified_gdf.index.to_list(), JOBS):
        chunks.append(c)
        # Parallel processing

    with tqdm(total=len(geographic_linear_densified_gdf['geometry']),
                 desc = 'Generating Transects',
                 unit=' Fault') as pbar:

            for i in range(len(chunks)):

                sub_chunk = chunks[i]

                files = [geographic_linear_densified_gdf.iloc[ch] for ch in sub_chunk]
                results=(Parallel (n_jobs=JOBS)(delayed(transecter)(linear_densified_gdf,geographic_linear_densified_gdf,
                                                               ii, length_param)
                                       for ii in chunks[i]))
                pbar.update(len(results))
                for res in results:                
                    chunk_results.append(res)
                    geoLengths.append(res[0][0])
                    segments.append(res[1])
                    temp_datas.append(res[2])
    return(geoLengths, segments, temp_datas)


def parallel_tReshaper(transect_gdf,transect_reshaper,riox_dem,dem_res):
    from pyproj import Geod
    geod = Geod('+a=1737400.0')
    
    JOBS = joblib.cpu_count()

    results = []
    Displacements = []
    Geoms = []
    chunks = []
    chunk_results=[]

    for c in chunk_creator(transect_gdf.index.to_list(), JOBS):
        chunks.append(c)
        # Parallel processing
        

    with tqdm(total=len(transect_gdf['geometry']),
                 desc = 'Generating Transects',
                 unit=' Fault') as pbar:

            for i in range(len(chunks)):

                sub_chunk = chunks[i]

                files = [transect_gdf.iloc[ch] for ch in sub_chunk]
                results=(Parallel (n_jobs=JOBS)(delayed(transect_reshaper)(transect_gdf.iloc[ii],ii,riox_dem,dem_res)
                                       for ii in sub_chunk))
                pbar.update(len(results))
                for res in results:                
                    chunk_results.append(res)
                    Displacements.append(res[0][0])
                    Geoms.append(res[1][0])
                    
    return(Displacements, Geoms)

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
import itertools


def transect_dataframe_creator(temp_datas):
    transect_stat_cols = ['Fault ID','Segment L(m)','geometry']
    transect_gdf = gpd.GeoDataFrame(columns=transect_stat_cols)#, dtype='int64')
    transect_gdf['Fault L(m)']=0
    with tqdm(total=len(temp_datas),
                 desc = 'Generating Transect Dataframe',
                 unit=' Transects') as pbar:
        for tmp in temp_datas:
            temp_gdf = gpd.GeoDataFrame.from_records(tmp)
            temp_gdf['TrID']=np.arange(0,len(temp_gdf))
            transect_gdf = pd.concat([transect_gdf,temp_gdf], ignore_index=True)#,join='inner')            
            transect_gdf.loc[transect_gdf['Fault ID']==tmp[-1]['Fault ID'],'Fault L(m)']=tmp[-1]['Segment L(m)']
            pbar.update(1)
    
    
    return(transect_gdf)

def transect_reshaper(sub_transect_gdf,idx,riox_dem,dem_res):
    TrDsps = []
    TrGeoms = []
    #for idx in range(len(sub_transect_gdf)):
    sub_df = sub_transect_gdf#.iloc[idx]
    geom = sub_df.geometry
    
    curr_len = geom.length
    corr_len = geom.length
    idx = int(sub_df['Fault ID'])
    
    sub_idx = int(sub_df['TrID'])
    fault_len = sub_df['Fault L(m)']            
    
    dem_profile, points = dem_profiler(riox_dem.squeeze(), geom, int(geom.length/dem_res))
    min_indxs = np.where(dem_profile == np.amin(dem_profile))
    min_indx =min_indxs[len(min_indxs)//2][0]
    min_val = dem_profile[min_indx]
    min_profs = list(split_after(dem_profile,lambda x: x==min_val, maxsplit=2))            
    #min_prof, minProfLength = FindMinLength(min_profs)    
    stop = math.ceil(geom.length)#.length[df_id])
    x = np.linspace(0, stop, num=len(dem_profile), endpoint=True)[0:len(min_profs[0])]
    #print(len(x))
    new_length = math.ceil(x[-1])
    new_geom = cut(geom,new_length)[0]
    cds3d = 0
    for i in range(len(points)):
        if i != len(points)-1:
            cds3d+=math.sqrt(pow(points[i].distance(points[i+1]),2)+pow(dem_profile[i+1]-dem_profile[i],2))
        else:
            cds3d+=math.sqrt(pow(points[-2].distance(points[-1]),2)+pow(dem_profile[-1]-dem_profile[-2],2))
    TrDsps.append(cds3d)
    TrGeoms.append(new_geom)
    #transect_gdf_upd.loc[transect_gdf_upd['Fault ID']==idx,'Tran L(m)']=FaultLen
    #transect_gdf_upd.loc[transect_gdf_upd['Fault ID']==idx,'TrDsp (m)']=cds3d
    #transect_gdf_upd.loc[transect_gdf_upd['Fault ID']==idx,'geometry']=new_geom
#transect_gdf_upd.loc[transect_gdf_upd['Fault ID']==idx,'TrDsp (m)']=TrDsps
#transect_gdf_upd.loc[transect_gdf_upd['Fault ID']==idx,'geometry']=TrGeoms

    return (TrDsps, TrGeoms, x)



### SEE https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1], upper[:-1]