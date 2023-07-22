from shapely.geometry import Polygon, Point,MultiPolygon
from arcgis.gis import GIS
from arcgis.geoenrichment import Country
from arcgis.geometry import Geometry
import pandas as pd
from shapely.ops import cascaded_union
import geopandas as gpd
from tqdm import tqdm
tqdm.pandas()
import shapely
import folium

import rasterio
import numpy as np
from rasterio import Affine, features
from shapely.geometry import mapping, shape
from shapely.ops import cascaded_union
from math import floor, ceil, sqrt

import pyproj

def convert_polygon_to_web_mercator_projection(poly):
    projected_crs = pyproj.CRS.from_epsg(3857)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    polygon_web_mercator = poly.__geo_interface__
    polygon_web_mercator["coordinates"] = [transformer.transform(x, y) for x, y in poly.exterior.coords]
    return Polygon(polygon_web_mercator["coordinates"])


from scipy.signal import fftconvolve

def gaussian_blur(in_array, gt, size):
    """Gaussian blur, returns tuple `(ar, gt2)` that have been expanded by `size`"""
    # expand in_array to fit edge of kernel; constant value is zero
    padded_array = np.pad(in_array, size, 'constant')
    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)
    # do the Gaussian blur
    ar = fftconvolve(padded_array, g, mode='full')
    # convolved increased size of array ('full' option); update geotransform
    gt2 = Affine(
        gt.a, gt.b, gt.xoff - (2 * size * gt.a),
        gt.d, gt.e, gt.yoff - (2 * size * gt.e))
    return ar, gt2

def median_polygon(shapes):
    #Convert the shapes from the WGS84 projection to the web mercator projection where 1 unit corresponds to 1 meter
    shapes_in_web_mercator=list(map(convert_polygon_to_web_mercator_projection,shapes))
    # We calculate the max shape as the union of all polygons in order to calculate a box that surrounds this polygon union
    max_shape = cascaded_union([shape(s) for s in shapes_in_web_mercator])
    minx, miny, maxx, maxy = max_shape.bounds
    # We define the grid reslution in meter of the raster
    dx = dy = 10  # grid resolution; this can be adjusted
    #We calculated the length of the raster in X and Y in unit of pixels
    lenx = dx * ( ceil(maxx / dx) - floor(minx / dx)  )
    leny = dy * (ceil(maxy / dy) - floor(miny / dy))
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    # We calculate the number of pixels that will be used for X and Y
    nx = int(lenx / dx)
    ny = int(leny / dy)
    # We create the affine transformation matrix which will equivalent to a matrix with no shear and Dx, Dy in transformation fields with certain translation
    gt = Affine(
        dx, 0.0, dx * floor(minx / dx),
        0.0, -dy, dy * ceil(maxy / dy))
    
    # After calculating the affine matrix we calculate the matrix of zeros
    pa = np.zeros((ny, nx), 'd')
    #And rasterize of all the polygons into the affine matrix. In here we'll project the polygons into that 2d space and count the number of pixels that each polygon cross
    for s in shapes_in_web_mercator:
        r = features.rasterize([s], (ny, nx), transform=gt)
        pa[r > 0] += 1
    # We normalise navluaes by dividing over the number of polygons. There we calculate the percentage of polygons that 
    # cross each pixel out of the set pass as paramter
    pa /= len(shapes)  # normalise values
    
    
    
    spa, sgt = gaussian_blur(pa, gt, 5)
    
    thresh = 0.5  # median
    pm = np.zeros(spa.shape, 'B')
    pm[spa > thresh] = 1
    
    poly_shapes = []
    for sh, val in features.shapes(pm, transform=sgt):
        if val == 1:
            poly_shapes.append(shape(sh))
    if not any(poly_shapes):
        raise ValueError("could not find any shapes")
    avg_poly = cascaded_union(poly_shapes)
    
    return avg_poly
    
