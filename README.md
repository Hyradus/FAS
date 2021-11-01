# QGIS-TransectScaler

These python scripts are ancillary material of 

Each script need to be executed in blocks (listed in each script and in the following documentation) using an IDE like Spyder, further development may include full automatization and optimizations.

## Requirements

### Software
* anaconda + provided environment
* QGIS
* qprof plugin for QGIS

### Data
* DEM
* Slope
* Aspect
* Orthoimage
* Faults shapefile

## Workflow

### First block

* Source data is loaded
* Faults shapefile in reprojected in the same CRS of the orthoimage
* Create a "processing" sub-folder in the root of the source data
* Each fault in the shapefile is densified in order to create equally distanced points that will be used to create transects in QGIS using qprof plugin.

### Second block

* Move to QGIS and load the densified shapefile available in the processing folder
* Under Processing, select *Transect* tool and set an initial value for the length greater than the average half-lenth of the grabens (may require some trials to get an overall value), left side, 90 degrees. Then save as shapefile.
* Back to the python script, start executing the second block with loading the transects shapefile (*transect_df_file*)
* Execute *transect_gdf_reshaper* to obtain a modified version of the transects shapefile. The *transect_reshaper* function, read both the original transects shapefile and the DEM file, compute the topography profile and cut the profile at the minimum value (that correspond to the bottom of the graben). 
* Load the reshaped shapefile in QGIS and check if there are transects with anomalies in the cut (e.g. still longer than the half-aperture, cutted to much, and so on). If present, go back in the python IDE and edit the *transect_reshaper* functions with custom values. This is necessary, since all the grabens and faults have not a perfect shape nor are aligned perfectly, so a second cut is performed by setting custom values, depending on each fault/transect. **NOTE that this step may requires some trials and errors steps**

### Third block
* Use qprof plugin and load the base DEM and the *transect reshaped shapefile*. Select *Layer with multiple profiles*, *label field*=Tran_id, *Line order field*="optional"
* Set a *line densify distance* lowest as possible (average 100)
* Click on *Read source data*
* Click on calculate statistics
* Click on create topographyc profile. Qgis will hang a bit, then just close the window with the profiles that are actually unusable due to visualization limits.
* Move on qProf Export tab
* Click on Create topographic profile and leave all settings as default
* Save the transect profiles as shapefile
* Back on the script, load the *transect reshaped profiles* shapefile
* Execute all the code up to FOURTH BLOCK. This will generate the final shapefile containing all the statistics.


### Fourth block

