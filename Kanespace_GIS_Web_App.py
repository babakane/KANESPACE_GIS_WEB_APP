# --- (Keep all imports the same, ensure pandas/geopandas are imported) ---
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
# import contextily as ctx # Contextily not explicitly used in mapping anymore
import matplotlib.pyplot as plt
import zipfile
# from io import BytesIO # Not used
import plotly.express as px
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import tempfile
from geopy.geocoders import Nominatim
import rasterio
from rasterio.plot import show
from sqlalchemy import create_engine, exc
import osmnx as ox
from shapely.geometry import shape, Point, LineString, Polygon # Added Point, LineString, Polygon for clarity if needed later
# from sklearn.cluster import DBSCAN # Removed direct import, use later if needed
# import tensorflow as tf  # Avoid top-level import if heavy
# from tensorflow import keras # Avoid top-level import if heavy
# import h3 # Not used
import leafmap.foliumap as leafmap
from datetime import datetime
import json
import logging # Import logging
import random # For geocoder user agent

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="KANESPACE Geospatial Analysis", # <--- Updated Title
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS (Keep as is)
# -------------------------------
st.markdown("""
    <style>
    :root {
        --primary: #4CAF50;
        --secondary: #2E7D32;
        --background: #f5f5f5;
    }
    .main {background-color: var(--background);}
    .stSelectbox:first-child {border: 2px solid var(--primary);}
    .stButton>button {background-color: var(--primary); color: white;}
    .stFileUploader>div>div>div>div {color: var(--primary);}
    .sidebar .sidebar-content {background-color: white;}
    [data-testid="stExpander"] {background: white; border-radius: 5px;}
    .dashboard-card {border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #fff;}
    .welcome-text { padding: 20px; background-color: #e8f5e9; border-left: 5px solid var(--primary); border-radius: 5px; margin-bottom: 20px;}
    .welcome-text h2 { color: var(--secondary); }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# App State Management (Keep as is)
# -------------------------------
if 'layers' not in st.session_state:
    st.session_state.layers = {} # Stores loaded data: {name: GeoDataFrame or {'type': 'raster', 'path': ...} or {'type': 'wms', ...}}
if 'active_layer' not in st.session_state:
    st.session_state.active_layer = None # Key (name) of the currently active layer in st.session_state.layers
if 'map_center' not in st.session_state:
    st.session_state.map_center = [51.5074, -0.1278] # Default to London center
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6 # Adjust default zoom
if 'dashboards' not in st.session_state:
    st.session_state.dashboards = {
        'Default Dashboard': {'type': 'General', 'components': ['Map View', 'Data Table']} # Add a default
    }
if 'temp_raster_paths' not in st.session_state:
    st.session_state.temp_raster_paths = set()

# -------------------------------
# Title and Header
# -------------------------------
# Use a less intrusive header, the welcome text will be prominent on Home page
st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);padding:15px 25px;border-radius:10px;margin-bottom:20px; color:white;">
        <h1 style="margin:0;font-size:2rem;">KANESPACE Geospatial Analysis 🌍</h1>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# Helper Functions
# -------------------------------
def cleanup_temp_files():
    """Removes temporary raster files created."""
    paths_to_remove = list(st.session_state.temp_raster_paths) # Iterate over a copy
    for path in paths_to_remove:
        try:
            if os.path.exists(path):
                os.unlink(path)
                logging.info(f"Cleaned up temporary file: {path}")
            st.session_state.temp_raster_paths.remove(path)
        except OSError as e:
            logging.error(f"Error removing temp file {path}: {e}")
        except KeyError:
            pass # Already removed

# *** NEW HELPER FUNCTION FOR DATAFRAME DISPLAY ***
def display_gdf_safely(gdf):
    """
    Prepares a GeoDataFrame for safe display in st.dataframe
    by converting geometry to WKT. Returns a standard DataFrame.
    Handles potential non-GDF inputs gracefully.
    """
    if isinstance(gdf, gpd.GeoDataFrame):
        df_display = gdf.copy()
        # Check if 'geometry' column exists before trying to convert
        if 'geometry' in df_display.columns:
            try:
                # Convert geometry to WKT (Well-Known Text) string representation
                df_display['geometry_wkt'] = df_display['geometry'].apply(lambda geom: geom.wkt if geom else None)
                # Drop the original geometry column for display
                df_display = df_display.drop(columns=['geometry'])
                # Optionally move WKT column to the end or beginning
                if 'geometry_wkt' in df_display.columns:
                    cols = [col for col in df_display.columns if col != 'geometry_wkt'] + ['geometry_wkt']
                    df_display = df_display[cols]
            except Exception as e:
                logging.warning(f"Could not convert geometry to WKT for display: {e}")
                # Fallback: drop geometry if conversion fails
                if 'geometry' in df_display.columns:
                     df_display = df_display.drop(columns=['geometry'])
        return pd.DataFrame(df_display) # Return as standard DataFrame
    elif isinstance(gdf, pd.DataFrame):
        # If it's already a Pandas DataFrame, return it as is
        return gdf
    else:
        # Handle other types if necessary, or return an empty DataFrame
        logging.warning(f"Input to display_gdf_safely was not a GeoDataFrame or DataFrame: {type(gdf)}")
        return pd.DataFrame() # Return empty DataFrame for safety

# -------------------------------
# AI & Machine Learning Utilities (Keep as is from previous revision)
# -------------------------------
def apply_ml_analysis(gdf, analysis_type):
    """Apply machine learning algorithms to spatial data"""
    try:
        if analysis_type == "Cluster Detection (DBSCAN)": # Match selectbox option text
            from sklearn.cluster import DBSCAN # Import heavy libraries locally

            # Ensure gdf has point geometries for simple clustering
            if not all(gdf.geom_type.isin(['Point', 'MultiPoint'])):
                 st.warning("Clustering currently supports Point geometries. Converting to centroids.")
                 try:
                     # Filter out potential None geometries before centroid calculation
                     valid_geoms = gdf.geometry.dropna()
                     if valid_geoms.empty:
                          st.error("No valid geometries found for centroid calculation.")
                          return None
                     coords = np.array([[geom.x, geom.y] for geom in valid_geoms.centroid if geom is not None])
                 except Exception as centroid_e:
                     st.error(f"Could not get centroids for clustering: {centroid_e}")
                     return None
            else:
                 # Filter out potential None geometries directly
                 valid_geoms = gdf.geometry.dropna()
                 if valid_geoms.empty:
                     st.error("No valid point geometries found for clustering.")
                     return None
                 coords = np.array([[geom.x, geom.y] for geom in valid_geoms if geom is not None])

            if coords.size == 0:
                 st.error("No valid coordinates found for clustering.")
                 return None

            # Simple parameter selection - might need adjustment based on data density/CRS
            # Consider scaling coordinates if not in projected CRS for eps meaningfulness
            db = DBSCAN(eps=0.1, min_samples=3).fit(coords)

            # Add cluster labels back - Handle potential index mismatch due to dropped NaNs
            result_gdf = gdf.loc[valid_geoms.index].copy() # Work with the subset that had valid geometry
            if len(db.labels_) == len(result_gdf):
                 result_gdf['cluster'] = db.labels_
            else:
                st.warning(f"Cluster label count ({len(db.labels_)}) doesn't match valid feature count ({len(result_gdf)}). Clustering might be incomplete.")
                # Attempt partial assignment if possible
                min_len = min(len(db.labels_), len(result_gdf))
                result_gdf['cluster'] = pd.Series(db.labels_[:min_len], index=result_gdf.index[:min_len])


            st.success(f"Clustering complete. Found {len(np.unique(db.labels_)) - (1 if -1 in db.labels_ else 0)} clusters (excluding noise points labeled -1).")
            # Return only the rows that were clustered
            return result_gdf

        elif analysis_type == "Land Use Prediction (Placeholder)":
            st.warning("Land use prediction requires pre-trained models and specific input features.")
            st.info("This is a placeholder for integrating a trained TensorFlow/Keras model.")
            # Example Placeholder Logic (replace with actual model loading and prediction)
            try:
                # import tensorflow as tf # Keep imports local
                # from tensorflow import keras
                # # model = keras.models.load_model('path/to/your/land_use_model.h5')
                # # Preprocess gdf features into expected model input format
                # # input_data = preprocess_for_landuse(gdf)
                # # predictions = model.predict(input_data)
                # # gdf['predicted_land_use'] = postprocess_predictions(predictions)
                temp_gdf = gdf.copy()
                temp_gdf['predicted_land_use'] = ['placeholder'] * len(temp_gdf) # Example placeholder
                st.success("Placeholder prediction applied.")
                return temp_gdf
            # except ImportError:
            #     st.error("TensorFlow/Keras not installed.")
            #     return None
            except Exception as e:
                st.error(f"Land use prediction failed: {str(e)}")
                return None
            # return None # Return None if placeholder/not implemented

        # Add other ML types here if needed

    except ImportError as e:
        st.error(f"ML analysis failed: Missing library - {e}")
        return None
    except Exception as e:
        st.error(f"ML analysis failed: {str(e)}")
        logging.error(f"ML analysis error: {e}", exc_info=True)
        return None
    return None # Default return if type not matched

# -------------------------------
# Data Integration & Management (Keep as is from previous revision)
# -------------------------------
# (Using the robust load_data and handle_data_integration from the previous answer)
def load_data(uploaded_file):
    """Universal data loader with format detection and error handling."""
    file_name = uploaded_file.name
    file_content = uploaded_file.getvalue() # Read content once
    logging.info(f"Attempting to load file: {file_name}")

    temp_dir = tempfile.TemporaryDirectory() # Create a temp dir for complex formats like shapefiles
    temp_file_path = os.path.join(temp_dir.name, file_name)

    try:
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        # Handle Zipped Shapefiles
        if file_name.lower().endswith('.zip'):
            logging.info(f"Detected zip file: {file_name}. Extracting...")
            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir.name)
            # Find the .shp file
            shp_files = [f for f in os.listdir(temp_dir.name) if f.lower().endswith('.shp')]
            if not shp_files:
                raise ValueError("No .shp file found in the zip archive.")
            file_to_read = os.path.join(temp_dir.name, shp_files[0])
            logging.info(f"Reading shapefile: {shp_files[0]}")
            gdf = gpd.read_file(file_to_read)
            temp_dir.cleanup() # Clean up after successful read
            return gdf

        # Handle other vector formats directly readable by GeoPandas
        elif file_name.lower().endswith(('.geojson', '.json', '.kml', '.gpkg', '.shp')):
            logging.info(f"Reading vector file: {file_name}")
            gdf = gpd.read_file(temp_file_path)
            # Basic geometry check
            if 'geometry' not in gdf.columns or gdf.geometry.isnull().all():
                 st.warning(f"File {file_name} loaded but no valid geometry column found/populated.")
                 # Optionally try to infer geometry if needed (complex, omitted for now)
            temp_dir.cleanup()
            return gdf

        # Handle Raster Data
        elif file_name.lower().endswith(('.tif', '.tiff', '.geotiff')):
            logging.info(f"Detected raster file: {file_name}")
            # Create a persistent temporary file for rasterio
            # Note: This file needs cleanup later using st.session_state.temp_raster_paths
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_raster:
                 tmp_raster.write(file_content)
                 raster_path = tmp_raster.name
            st.session_state.temp_raster_paths.add(raster_path) # Register for cleanup
            logging.info(f"Stored raster temporarily at: {raster_path}")
            temp_dir.cleanup() # Clean up the directory, keep the named temp file
            return {'type': 'raster', 'path': raster_path, 'name': file_name}

        # Handle CSV with potential coordinates
        elif file_name.lower().endswith('.csv'):
            logging.info(f"Reading CSV file: {file_name}")
            df = pd.read_csv(temp_file_path)
            temp_dir.cleanup()
            # Try common lat/lon column names
            lat_col, lon_col = None, None
            common_lats = ['lat', 'latitude', 'Lat', 'Latitude', 'y', 'Y']
            common_lons = ['lon', 'lng', 'longitude', 'Lon', 'Lng', 'Longitude', 'x', 'X']
            for col in common_lats:
                if col in df.columns:
                    lat_col = col
                    break
            for col in common_lons:
                if col in df.columns:
                    lon_col = col
                    break

            if lat_col and lon_col:
                logging.info(f"Found coordinates in CSV: '{lon_col}', '{lat_col}'")
                # Convert to numeric, coercing errors
                df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                # Drop rows where conversion failed
                df.dropna(subset=[lon_col, lat_col], inplace=True)
                if not df.empty:
                     gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326") # Assume WGS84
                     logging.info(f"Converted CSV to GeoDataFrame with {len(gdf)} points.")
                     return gdf
                else:
                     st.warning(f"CSV {file_name}: Found coordinate columns but no valid numeric coordinate pairs after conversion.")
                     return df # Return original df if conversion fails or results in empty
            else:
                 logging.info(f"CSV {file_name}: No standard coordinate columns found.")
                 return df # Return as plain DataFrame

        # Handle Excel files (requires openpyxl)
        elif file_name.lower().endswith(('.xls', '.xlsx')):
             logging.info(f"Reading Excel file: {file_name}")
             try:
                 df = pd.read_excel(temp_file_path)
                 temp_dir.cleanup()
                 # Add logic here to check for lat/lon columns similar to CSV if needed
                 return df # Return as plain DataFrame for now
             except ImportError:
                  temp_dir.cleanup()
                  st.error("Reading Excel files requires the 'openpyxl' library. Please install it (`pip install openpyxl`).")
                  return None

        else:
             temp_dir.cleanup()
             st.warning(f"Unsupported file type: {file_name}")
             return None

    except Exception as e:
        logging.error(f"Error loading file {file_name}: {e}", exc_info=True)
        st.error(f"Failed to load {file_name}: {str(e)}")
        temp_dir.cleanup() # Ensure cleanup on error
        return None

def handle_data_integration():
    """Enhanced data integration panel"""
    with st.expander("🧩 Data Integration Hub", expanded=True):
        tab1, tab2, tab3 = st.tabs(["File Upload", "Database Connect", "Web Services"])

        with tab1:
            uploaded_files = st.file_uploader("Upload data files",
                                            type=["geojson", "json", "kml", "shp", "zip",
                                                  "tif", "tiff", "csv", "gpkg", "xlsx"],
                                            accept_multiple_files=True, key="fileuploader")

            if uploaded_files:
                new_layer_loaded = False
                for file in uploaded_files:
                    # Check if layer *name* already exists
                    if file.name not in st.session_state.layers:
                        with st.spinner(f"Loading {file.name}..."):
                            data = load_data(file)
                        if data is not None:
                            st.session_state.layers[file.name] = data
                            st.success(f"Successfully loaded: {file.name}")
                            st.session_state.active_layer = file.name # Set the last loaded as active
                            new_layer_loaded = True
                        else:
                             # Error message handled within load_data
                             pass
                    else:
                         st.info(f"Layer '{file.name}' already loaded. To reload, clear it first.")

        with tab2:
            st.subheader("Connect to Database")
            db_type = st.selectbox("Database Type", ["PostGIS", "MongoDB (Not Implemented)", "ArcGIS (Not Implemented)"], key="db_type")
            conn_str = st.text_input("Connection String (e.g., postgresql://user:password@host:port/dbname)", key="db_conn_str")

            if db_type == "PostGIS":
                if st.button("Connect & List Tables", key="db_connect"):
                    if conn_str:
                        try:
                            engine = create_engine(conn_str)
                            # Query for tables with geometry columns OR common public tables (adjust schema if needed)
                            query = """
                                SELECT table_schema || '.' || f_table_name as qualified_name
                                FROM geometry_columns
                                UNION
                                SELECT table_schema || '.' || table_name as qualified_name
                                FROM information_schema.tables
                                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                                ORDER BY qualified_name;
                            """
                            with engine.connect() as connection:
                                tables_df = pd.read_sql(query, connection)

                            if not tables_df.empty:
                                st.session_state.db_tables = tables_df['qualified_name'].tolist()
                                st.info(f"Found {len(st.session_state.db_tables)} tables. Select below.")
                            else:
                                st.warning("No tables found or accessible.")
                                if 'db_tables' in st.session_state: del st.session_state.db_tables

                        except exc.OperationalError as e:
                            st.error(f"PostGIS Connection failed: Check connection string and credentials. ({e})")
                        except Exception as e:
                            st.error(f"Database error: {str(e)}")
                            logging.error("DB Connection Error", exc_info=True)
                    else:
                        st.warning("Please enter a connection string.")

                if 'db_tables' in st.session_state and st.session_state.db_tables:
                    selected_table = st.selectbox("Select table to load", st.session_state.db_tables, key="db_table_select")
                    load_table_button = st.button(f"Load Table '{selected_table}'", key="db_load_table")

                    if load_table_button and selected_table:
                        layer_name = f"db_{selected_table}"
                        if layer_name not in st.session_state.layers:
                             with st.spinner(f"Loading {selected_table}..."):
                                try:
                                    engine = create_engine(conn_str) # Recreate engine instance safely
                                    # Try loading with GeoPandas directly
                                    gdf = gpd.read_postgis(f"SELECT * FROM {selected_table}", engine, geom_col='geometry') # Adjust geom_col if needed
                                    st.session_state.layers[layer_name] = gdf
                                    st.success(f"Loaded '{selected_table}' ({len(gdf)} records)")
                                    st.session_state.active_layer = layer_name
                                except Exception as e:
                                    st.error(f"Failed to load table '{selected_table}': {e}")
                                    logging.error(f"DB Table Load Error for {selected_table}", exc_info=True)
                        else:
                            st.info(f"Layer '{layer_name}' already loaded.")

            else:
                 st.info(f"{db_type} connection not implemented yet.")


        with tab3:
            st.subheader("Connect to Web Services")
            service_type = st.selectbox("Service Type", ["WMS", "WFS (Not Implemented)", "ArcGIS REST (Not Implemented)"], key="ws_type")
            url = st.text_input("Service URL (e.g., WMS GetCapabilities URL)", key="ws_url")
            layer_name_input = st.text_input("Layer Name(s) (comma-separated for WMS)", key="ws_layer_name")

            if service_type == "WMS":
                if st.button("Add WMS Layer", key="ws_add"):
                    if url and layer_name_input:
                         layer_id = f"wms_{layer_name_input.split(',')[0]}" # Use first layer name for ID
                         if layer_id not in st.session_state.layers:
                              st.session_state.layers[layer_id] = {
                                  'type': 'wms',
                                  'url': url,
                                  'layers': layer_name_input, # Store original comma-separated list
                                  'name': layer_id # Store name for display
                              }
                              st.success(f"WMS layer '{layer_name_input}' added. Select in map overlays.")
                              # WMS layers don't become 'active' in the same way GDFs do
                         else:
                              st.info(f"WMS layer '{layer_id}' already added.")
                    else:
                         st.warning("Please provide both WMS URL and Layer Name(s).")
            else:
                 st.info(f"{service_type} connection not implemented yet.")

# -------------------------------
# Advanced Visualization Engine (Keep as is from previous revision)
# -------------------------------
# (Using the robust create_interactive_map from the previous answer)
def create_interactive_map():
    """Enhanced mapping with multiple view options and layer control."""
    st.subheader("🗺️ Interactive Map Visualization")
    try:
        map_container = st.container() # Use a container for map display

        with st.expander("Map Controls & Layers", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                 map_type = st.selectbox("Map Type", ["Interactive (Leafmap)", "3D (Pydeck)", "Heatmap (Leafmap)"], key="map_type_select") # Removed Time Series for now
                 basemap = st.selectbox("Basemap", ["OpenStreetMap", "Satellite", "Terrain", "CartoDB Positron", "CartoDB DarkMatter"], key="basemap_select")

            with col2:
                 if st.session_state.layers:
                      available_layers = list(st.session_state.layers.keys())
                      # Ensure active_layer is valid if it exists
                      current_active = st.session_state.active_layer if st.session_state.active_layer in available_layers else None
                      # Handle case where available_layers might be empty after filtering/deletion
                      active_layer_index = 0
                      if current_active and available_layers:
                          try:
                              active_layer_index = available_layers.index(current_active)
                          except ValueError: # If active layer was deleted but state not updated yet
                              current_active = None # Reset current_active
                              st.session_state.active_layer = None

                      selected_active = st.selectbox("Set Active Layer (for analysis/details)", available_layers,
                                                     index=active_layer_index,
                                                     key="active_layer_selector_map")
                      if selected_active != st.session_state.active_layer:
                           st.session_state.active_layer = selected_active

                      # Set default overlays: active layer if exists, else all layers
                      default_overlays = []
                      if st.session_state.active_layer and st.session_state.active_layer in available_layers:
                           default_overlays = [st.session_state.active_layer]
                      elif available_layers: # If no active layer, default to showing all
                           default_overlays = available_layers

                      overlay_layers_selected = st.multiselect("Visible Overlay Layers",
                                                               available_layers,
                                                               default=default_overlays,
                                                               key="map_overlays")
                 else:
                      st.info("Load data to see overlay options.")
                      overlay_layers_selected = []

                 # Add map-type specific controls here if needed (e.g., heatmap radius)
                 if map_type == "Heatmap (Leafmap)":
                      heatmap_radius = st.slider("Heatmap Radius", 5, 50, 15, key="heatmap_radius")
                      heatmap_blur = st.slider("Heatmap Blur", 5, 50, 15, key="heatmap_blur")


        with map_container:
            if not overlay_layers_selected:
                 st.warning("No data loaded or selected to display on the map.")
                 # Display a default empty map maybe?
                 m = leafmap.Map(center=st.session_state.map_center, zoom=st.session_state.map_zoom)
                 m.add_basemap(basemap.replace(" ", ""))
                 m.to_streamlit(height=700, width='100%')
                 return

            active_gdf = None
            if st.session_state.active_layer and st.session_state.active_layer in st.session_state.layers:
                layer_data = st.session_state.layers[st.session_state.active_layer]
                if isinstance(layer_data, gpd.GeoDataFrame):
                    active_gdf = layer_data

            # Calculate map center/zoom based on selected layers
            map_bounds = None
            all_bounds = []
            for name in overlay_layers_selected:
                layer = st.session_state.layers.get(name)
                if isinstance(layer, gpd.GeoDataFrame) and not layer.empty:
                    try:
                         # Filter out invalid or empty geometries before calculating bounds
                         valid_geoms = layer.geometry.dropna()
                         valid_geoms = valid_geoms[~valid_geoms.is_empty]
                         if not valid_geoms.empty:
                              temp_gdf = gpd.GeoDataFrame(geometry=valid_geoms, crs=layer.crs)
                              all_bounds.append(temp_gdf.total_bounds)
                    except Exception as e:
                         logging.warning(f"Could not get bounds for layer {name}: {e}")
                # Add logic here to get bounds for raster/WMS if possible/needed (complex)

            if all_bounds:
                 # Aggregate bounds: min_lon, min_lat, max_lon, max_lat
                 min_lon = min(b[0] for b in all_bounds)
                 min_lat = min(b[1] for b in all_bounds)
                 max_lon = max(b[2] for b in all_bounds)
                 max_lat = max(b[3] for b in all_bounds)
                 # Basic check for valid bounds
                 if min_lon < max_lon and min_lat < max_lat:
                     map_bounds = [[min_lat, min_lon], [max_lat, max_lon]] # Folium/Leafmap format [[south, west], [north, east]]
                     # Calculate center dynamically
                     st.session_state.map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
                 else:
                      logging.warning("Calculated invalid bounds from selected layers.")


            # --- Render selected map type ---
            if map_type == "Interactive (Leafmap)":
                m = leafmap.Map(center=st.session_state.map_center,
                              zoom=st.session_state.map_zoom, # Initial zoom, fit_bounds will override if possible
                              layers_control=True) # Add layer control toggle

                # Add basemap using leafmap's convention
                m.add_basemap(basemap.replace(" ", "")) # e.g., "CartoDB Positron" -> "CartoDBPositron"

                # Add selected overlay layers
                layer_added = False
                for name in overlay_layers_selected:
                    layer = st.session_state.layers.get(name)
                    if isinstance(layer, gpd.GeoDataFrame):
                         if not layer.empty:
                            try:
                                 m.add_gdf(layer, layer_name=name, zoom_to_layer=False) # Add layer without auto-zooming each time
                                 layer_added = True
                            except Exception as e:
                                 st.error(f"Error adding GeoDataFrame layer '{name}' to map: {e}")
                                 logging.error(f"Map GDF add error ({name})", exc_info=True)
                         # else: # Don't clutter with info messages for empty layers
                              # st.info(f"Layer '{name}' is empty, skipping map display.")
                    elif isinstance(layer, dict) and layer.get('type') == 'raster':
                        try:
                             if os.path.exists(layer['path']):
                                 m.add_raster(layer['path'], name=layer.get('name', name), band=1) # Add first band by default
                                 layer_added = True
                             else:
                                 st.warning(f"Raster file path for '{name}' not found: {layer.get('path', 'N/A')}. Skipping.")
                        except Exception as e:
                            st.error(f"Error adding raster layer '{name}' to map: {e}")
                            logging.error(f"Map Raster add error ({name})", exc_info=True)
                    elif isinstance(layer, dict) and layer.get('type') == 'wms':
                         try:
                             m.add_wms_layer(url=layer['url'], layers=layer['layers'], name=layer.get('name', name), format='image/png', transparent=True)
                             layer_added = True
                         except Exception as e:
                             st.error(f"Error adding WMS layer '{name}' to map: {e}")
                             logging.error(f"Map WMS add error ({name})", exc_info=True)

                if layer_added and map_bounds:
                     try:
                          m.fit_bounds(map_bounds) # Fit map to extent of selected layers
                     except Exception as e:
                          st.warning(f"Could not fit map bounds automatically: {e}")
                elif not layer_added:
                     st.warning("No valid layers selected or available to display on the map.")

                # Render the map
                map_output = m.to_streamlit(height=700, width='100%')

            elif map_type == "3D (Pydeck)":
                if active_gdf is not None and not active_gdf.empty:
                    try:
                         # Ensure WGS84 for Pydeck display
                         if active_gdf.crs and active_gdf.crs.to_epsg() != 4326:
                              gdf_display = active_gdf.to_crs(epsg=4326)
                         else:
                              gdf_display = active_gdf.copy() # Work on a copy

                         # Ensure geometries are valid before proceeding
                         gdf_display = gdf_display[gdf_display.geometry.is_valid & ~gdf_display.geometry.is_empty & gdf_display.geometry.notna()]

                         if gdf_display.empty:
                              st.warning("Active layer has no valid geometries for 3D view after preparing for display.")
                              return

                         center_lon = gdf_display.geometry.centroid.x.mean()
                         center_lat = gdf_display.geometry.centroid.y.mean()

                         # Basic elevation based on an attribute or constant
                         elevation_col = None
                         numeric_cols = gdf_display.select_dtypes(include=np.number).columns.tolist()
                         if 'elevation' in numeric_cols:
                             elevation_col = 'elevation'
                         elif 'height' in numeric_cols:
                             elevation_col = 'height'
                         # Add a small constant elevation if no suitable column
                         if not elevation_col:
                              gdf_display['elevation_3d'] = 10 # Default small elevation
                              elevation_col = 'elevation_3d'
                         else:
                              # Handle potential NaNs in existing elevation column
                              gdf_display[elevation_col] = pd.to_numeric(gdf_display[elevation_col], errors='coerce').fillna(1) # Fill NaN with 1 or 0

                         view_state = pdk.ViewState(
                              latitude=center_lat,
                              longitude=center_lon,
                              zoom=11, # Adjust initial zoom as needed
                              pitch=45,
                              bearing=0
                         )

                         # Choose layer type based on geometry (use first valid geom type)
                         geom_type = gdf_display.geom_type.iloc[0]
                         pdk_layer = None
                         layer_props_common = {
                              "data": gdf_display,
                              "pickable": True,
                              "auto_highlight": True,
                         }

                         if geom_type in ['Point', 'MultiPoint']:
                              pdk_layer = pdk.Layer(
                                   "ScatterplotLayer",
                                   **layer_props_common,
                                   get_position="geometry.coordinates",
                                   get_color=[255, 0, 0, 160], # Red points
                                   get_radius=50, # Radius in meters
                              )
                         elif geom_type in ['LineString', 'MultiLineString']:
                               pdk_layer = pdk.Layer(
                                   "PathLayer",
                                    **layer_props_common,
                                   get_path="geometry.coordinates",
                                   get_color=[0, 0, 255, 160], # Blue lines
                                   get_width=5, # Width in pixels
                               )
                         elif geom_type in ['Polygon', 'MultiPolygon']:
                              pdk_layer = pdk.Layer(
                                   "GeoJsonLayer", # GeoJsonLayer handles Polygons well
                                    **layer_props_common,
                                   opacity=0.8,
                                   stroked=True, # Draw borders
                                   filled=True,
                                   extruded=True, # Make it 3D
                                   wireframe=True,
                                   get_elevation=f"properties.{elevation_col}", # Use property for elevation
                                   get_fill_color=[255, 0, 0, 160], # Red fill
                                   get_line_color=[0, 0, 0], # Black lines
                              )

                         if pdk_layer:
                             # Prepare tooltip data safely - convert non-serializable types if needed
                             tooltip_df = gdf_display.drop(columns=['geometry'], errors='ignore').copy()
                             safe_tooltip_cols = {}
                             for col in tooltip_df.columns:
                                  # Simple check for common non-serializable types (add more if needed)
                                 if tooltip_df[col].dtype == 'datetime64[ns]':
                                     safe_tooltip_cols[col] = tooltip_df[col].astype(str)
                                 else:
                                     safe_tooltip_cols[col] = tooltip_df[col]
                             safe_tooltip_df = pd.DataFrame(safe_tooltip_cols)
                             pdk_layer.data = safe_tooltip_df # Update layer data with safe tooltip columns

                             tooltip = {"html": "<b>Feature Details:</b><br/>"}
                             for col in safe_tooltip_df.columns:
                                 tooltip["html"] += f"<b>{col}:</b> {{properties.{col}}}<br/>"

                             st.pydeck_chart(pdk.Deck(
                                 layers=[pdk_layer],
                                 initial_view_state=view_state,
                                 map_style='mapbox://styles/mapbox/satellite-v9', # Use satellite for 3D context
                                 tooltip=tooltip
                             ))
                         else:
                              st.warning(f"Geometry type '{geom_type}' not directly supported by this 3D setup.")

                    except Exception as e:
                         st.error(f"3D map creation failed: {e}")
                         logging.error("3D Map Error", exc_info=True)
                else:
                     st.warning("Select an active vector layer with geometries to use the 3D map.")


            elif map_type == "Heatmap (Leafmap)":
                 # Check if active layer is suitable
                 if active_gdf is not None and not active_gdf.empty:
                      point_gdf = active_gdf[active_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()
                      point_gdf = point_gdf[point_gdf.geometry.notna() & ~point_gdf.geometry.is_empty]

                      if not point_gdf.empty:
                           m = leafmap.Map(center=st.session_state.map_center, zoom=st.session_state.map_zoom)
                           m.add_basemap(basemap.replace(" ", ""))

                           # Extract coordinates for heatmap
                           coords = [[p.y, p.x] for p in point_gdf.geometry]

                           if coords:
                               m.add_heatmap(coords,
                                             radius=heatmap_radius,
                                             blur=heatmap_blur,
                                             name="Heatmap")
                               if map_bounds:
                                   try: m.fit_bounds(map_bounds)
                                   except: pass # Ignore fit_bounds errors for heatmap
                               m.to_streamlit(height=700, width='100%')
                           else:
                                st.warning("No valid point coordinates found in the active layer for heatmap.")
                      else:
                           st.warning("Active layer does not contain valid Point geometries for heatmap.")
                 else:
                      st.warning("Select an active layer containing Point geometries to generate a heatmap.")


    except Exception as e:
        st.error(f"Map creation failed unexpectedly: {str(e)}")
        logging.error("Map Creation Error", exc_info=True)


# -------------------------------
# Spatial Analysis Techniques (Keep as is from previous revision)
# -------------------------------
# (Using the robust perform_advanced_analysis from the previous answer)
def perform_advanced_analysis():
    """Enhanced spatial analysis panel"""
    st.subheader("🔍 Advanced Spatial Analysis")

    # Ensure active layer is selected and is a GeoDataFrame for most analyses
    active_layer_data = None
    if st.session_state.active_layer and st.session_state.active_layer in st.session_state.layers:
        layer_candidate = st.session_state.layers[st.session_state.active_layer]
        if isinstance(layer_candidate, gpd.GeoDataFrame):
            active_layer_data = layer_candidate
        else:
            st.warning(f"Selected active layer '{st.session_state.active_layer}' is not a vector layer suitable for most analyses.")
    # Removed the redundant else block here, handled by checks within tabs

    tab1, tab2, tab3, tab4 = st.tabs([
        "Geostatistics",
        "Network Analysis",
        "AI/ML",
        "Environmental"
    ])

    with tab1:
        st.markdown("**Geostatistics & Spatial Patterns**")
        analysis_type = st.selectbox("Analysis Type", [
            "Select...",
            "Spatial Autocorrelation (Moran's I)",
            "Hotspot Analysis (Getis-Ord Gi*) - Placeholder",
            "Interpolation (IDW) - Placeholder"
        ], key="geostat_type")

        # Only proceed if a valid analysis is selected AND active data is suitable
        if analysis_type != "Select...":
            if active_layer_data is not None and not active_layer_data.empty:
                gdf = active_layer_data # Use the checked variable

                if analysis_type == "Spatial Autocorrelation (Moran's I)":
                    numeric_cols = gdf.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select numeric column for analysis", numeric_cols, key="moran_col")
                        if st.button("Run Moran's I", key="run_moran"):
                            with st.spinner("Calculating spatial weights and Moran's I..."):
                                try:
                                    from esda.moran import Moran
                                    from libpysal.weights import Queen

                                    # Ensure valid geometries and non-empty gdf, drop NaNs for selected column
                                    gdf_clean = gdf.dropna(subset=[selected_col, 'geometry'])
                                    gdf_clean = gdf_clean[gdf_clean.geometry.is_valid & ~gdf_clean.geometry.is_empty]

                                    if gdf_clean.empty:
                                         st.error("No valid data after removing NaNs/invalid geometries for analysis.")
                                         # Use return instead of st.stop()
                                         return

                                    # Calculate spatial weights (Queen contiguity)
                                    w = Queen.from_dataframe(gdf_clean, use_index=False) # Avoid potential index issues
                                    w.transform = 'r' # Row-standardize weights

                                    moran = Moran(gdf_clean[selected_col].values, w) # Use .values for numpy array input
                                    st.success(f"Moran's I calculation complete:")
                                    st.metric(label="Moran's I Statistic", value=f"{moran.I:.4f}")
                                    st.metric(label="P-value (Simulation)", value=f"{moran.p_sim:.4f}")
                                    # Interpretation help
                                    if moran.p_sim < 0.05:
                                        if moran.I > 0:
                                            st.info("Result suggests positive spatial autocorrelation (clustering of similar values).")
                                        elif moran.I < 0:
                                            st.info("Result suggests negative spatial autocorrelation (dispersion of similar values).")
                                        # Moran's I is typically compared to expected value E[I] = -1/(n-1)
                                        # A value near 0 doesn't strongly indicate randomness unless E[I] is also near 0 (large n)
                                        # else:
                                        #      st.info("Result suggests random spatial pattern (Moran's I near E[I]).")
                                    else:
                                        st.info("Result is not statistically significant (p >= 0.05), cannot reject null hypothesis of spatial randomness.")

                                except ImportError:
                                     st.error("Analysis failed: Requires 'esda' and 'libpysal'. Install with `pip install esda libpysal`")
                                except Exception as e:
                                    st.error(f"Moran's I analysis failed: {str(e)}")
                                    logging.error("Moran's I Error", exc_info=True)
                    else:
                        st.warning("Active layer has no numeric columns suitable for Moran's I analysis.")
                elif analysis_type.startswith("Hotspot") or analysis_type.startswith("Interpolation"):
                     st.info(f"{analysis_type} is a placeholder for future implementation.")

            else: # If analysis selected but no suitable active data
                st.warning("Load or select an active vector layer with valid geometries first.")


    with tab2:
        st.markdown("**Network Analysis (using OSMnx)**")
        location_query = st.text_input("Enter location for network analysis (e.g., 'Manhattan, New York, USA')", "London, UK", key="network_location")
        network_type = st.selectbox("Network Type", ['drive', 'walk', 'bike', 'all_private', 'all'], index=0, key="network_type")
        analysis_option = st.selectbox("Analysis Action", [
            "Fetch and Display Network",
            "Calculate Basic Stats",
            "Service Areas (Isochrones) - Placeholder",
            "Optimal Routing - Placeholder"
        ], key="network_action")

        if st.button("Run Network Analysis", key="run_network"):
            if location_query:
                with st.spinner(f"Fetching '{network_type}' network for '{location_query}'..."):
                    try:
                        G = ox.graph_from_place(location_query, network_type=network_type)
                        st.success(f"Network graph fetched successfully ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges).")

                        if analysis_option == "Fetch and Display Network":
                             # Project graph for accurate plotting if needed (optional)
                             # G_proj = ox.project_graph(G)
                             # fig, ax = ox.plot_graph(G_proj, show=False, close=False, node_size=0)
                             fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0) # Simpler plot without projection
                             st.pyplot(fig)

                        elif analysis_option == "Calculate Basic Stats":
                             G_proj = ox.project_graph(G) # Stats often need projection
                             stats = ox.basic_stats(G_proj)
                             # Filter for display - convert non-basic types like lists to strings
                             stats_display = {}
                             for k, v in stats.items():
                                 if isinstance(v, (int, float, str)):
                                     stats_display[k] = v
                                 elif isinstance(v, list):
                                      stats_display[k] = ', '.join(map(str, v)) # Convert list to string
                                 # Add other conversions if needed
                             st.json(stats_display)


                        elif analysis_option.startswith("Service Areas") or analysis_option.startswith("Optimal Routing"):
                              st.info(f"{analysis_option} is a placeholder.")

                    except Exception as e:
                        st.error(f"Network analysis failed: {str(e)}")
                        logging.error("OSMnx Network Error", exc_info=True)
            else:
                st.warning("Please enter a location for network analysis.")

    with tab3:
        st.markdown("**AI / Machine Learning Analysis**")
        ml_type = st.selectbox("Select ML Analysis", [
            "Select...",
            "Cluster Detection (DBSCAN)",
            "Land Use Prediction (Placeholder)",
            "Risk Assessment (Placeholder)"
        ], key="ml_analysis_type")

        if ml_type != "Select...":
            if active_layer_data is not None and not active_layer_data.empty:
                gdf = active_layer_data # Use checked data
                if st.button(f"Apply {ml_type}", key="run_ml"):
                    with st.spinner(f"Running {ml_type} analysis..."):
                         result_gdf = apply_ml_analysis(gdf.copy(), ml_type) # Work on a copy

                         if result_gdf is not None:
                             # Generate a unique name for the result layer
                             base_name = st.session_state.active_layer or "analysis"
                             timestamp = datetime.now().strftime("%H%M%S")
                             layer_name = f"ml_{ml_type.split(' ')[0]}_{base_name}_{timestamp}"
                             st.session_state.layers[layer_name] = result_gdf
                             st.success(f"ML analysis complete! Results added as layer: '{layer_name}'.")
                             st.session_state.active_layer = layer_name # Make the result active
                             st.info("Result layer is now active. View it on the map or analyze further.")
                             # Optionally show a preview of results using the safe display function
                             st.dataframe(display_gdf_safely(result_gdf.head()), use_container_width=True)
                             # Trigger map refresh if needed
                         # else: Error handled in apply_ml_analysis
            else:
                 st.warning("Load or select an active vector layer with valid geometries first.")


    with tab4:
        st.markdown("**Environmental Modeling**")
        env_analysis = st.selectbox("Environmental Analysis Type", [
            "NDVI Calculation (Raster Placeholder)",
            "Watershed Delineation (Placeholder)",
            "Habitat Suitability (Placeholder)"
        ], key="env_analysis_type")

        if st.button("Run Environmental Analysis", key="run_env"):
            if env_analysis == "NDVI Calculation (Raster Placeholder)":
                 st.warning("NDVI requires specific raster bands (Red, NIR). Select a suitable raster layer.")
                 # Add logic here to find active raster layer, check bands, calculate NDVI
                 active_raster_path = None
                 if st.session_state.active_layer and isinstance(st.session_state.layers.get(st.session_state.active_layer), dict):
                      layer_info = st.session_state.layers[st.session_state.active_layer]
                      if layer_info.get('type') == 'raster' and os.path.exists(layer_info.get('path')):
                           active_raster_path = layer_info['path']

                 if active_raster_path:
                      st.info(f"Placeholder: Would perform NDVI on {active_raster_path}")
                      # try:
                      #    with rasterio.open(active_raster_path) as src:
                      #        # Check band count, assume bands (e.g., Sentinel: Red=B4, NIR=B8)
                      #        # red = src.read(4).astype(float)
                      #        # nir = src.read(8).astype(float)
                      #        # np.seterr(divide='ignore', invalid='ignore') # Ignore division by zero
                      #        # ndvi = (nir - red) / (nir + red)
                      #        # ... save or display ndvi ...
                      # except Exception as e:
                      #    st.error(f"NDVI calculation failed: {e}")
                 else:
                      st.warning("No active and valid raster layer selected.")

            else:
                 st.info(f"{env_analysis} is a placeholder for future implementation.")


# -------------------------------
# Custom Dashboard System (Apply safe display to table)
# -------------------------------
def manage_dashboards():
    """Dashboard creation and management with improved robustness."""
    st.subheader("📊 Custom Dashboards")

    tab1, tab2 = st.tabs(["View & Manage Dashboards", "Create New Dashboard"])

    with tab1:
        if not st.session_state.dashboards:
            st.info("No dashboards created yet. Use the 'Create New Dashboard' tab.")
            return

        dashboard_keys = list(st.session_state.dashboards.keys())
        selected_dashboard_name = st.selectbox(
            "Select Dashboard to View",
            dashboard_keys,
            key="view_dashboard_select"
        )

        if selected_dashboard_name and selected_dashboard_name in st.session_state.dashboards:
            dashboard = st.session_state.dashboards[selected_dashboard_name]
            st.markdown(f"### {selected_dashboard_name} ({dashboard.get('type', 'General')})")

            num_components = len(dashboard['components'])
            if num_components == 0:
                st.info("This dashboard has no components.")
                return

            cols = st.columns(min(num_components, 2))

            for i, component_name in enumerate(dashboard['components']):
                col_index = i % min(num_components, 2) # Assign to columns cyclically
                with cols[col_index]:
                    # Use st.container to group elements visually within the card effect
                    with st.container():
                        st.markdown(f"<div class='dashboard-card'><h5>{component_name}</h5></div>", unsafe_allow_html=True)
                        # Render component content within the card
                        try:
                            if component_name == "Map View":
                                if st.session_state.layers:
                                    # Determine layer to show (active or first)
                                    layer_to_show_name = st.session_state.active_layer
                                    available_layers = list(st.session_state.layers.keys())
                                    if not layer_to_show_name or layer_to_show_name not in available_layers:
                                        if available_layers:
                                             layer_to_show_name = available_layers[0]
                                        else:
                                             layer_to_show_name = None

                                    if layer_to_show_name:
                                        layer_data = st.session_state.layers[layer_to_show_name]
                                        m = leafmap.Map(center=st.session_state.map_center, zoom=6) # Smaller default zoom
                                        m.add_basemap("OpenStreetMap") # Simple base map

                                        if isinstance(layer_data, gpd.GeoDataFrame):
                                            m.add_gdf(layer_data, layer_name=layer_to_show_name)
                                        elif isinstance(layer_data, dict) and layer_data.get('type') == 'raster':
                                            if os.path.exists(layer_data['path']):
                                                m.add_raster(layer_data['path'], name=layer_to_show_name, band=1)
                                        elif isinstance(layer_data, dict) and layer_data.get('type') == 'wms':
                                             m.add_wms_layer(url=layer_data['url'], layers=layer_data['layers'], name=layer_to_show_name, format='image/png', transparent=True)
                                        # Fit bounds if possible (might be slow for dashboards)
                                        # if isinstance(layer_data, gpd.GeoDataFrame) and not layer_data.empty:
                                        #     try: m.fit_bounds(layer_data.total_bounds[[1,0,3,2]].tolist())
                                        #     except: pass
                                        m.to_streamlit(height=300) # Smaller map
                                    else:
                                        st.info("No layers available for Map View.")
                                else:
                                    st.info("Load data to display Map View.")

                            elif component_name == "Data Table":
                                active_layer_name = st.session_state.active_layer
                                if active_layer_name and active_layer_name in st.session_state.layers:
                                    layer_data = st.session_state.layers[active_layer_name]
                                    if isinstance(layer_data, (gpd.GeoDataFrame, pd.DataFrame)):
                                        # *** USE THE HELPER FUNCTION HERE ***
                                        st.dataframe(display_gdf_safely(layer_data.head()), use_container_width=True)
                                    else:
                                        st.info(f"Active layer '{active_layer_name}' is not tabular.")
                                elif st.session_state.layers:
                                     st.info("Select an active layer to view its data table.")
                                else:
                                    st.info("Load data to display Data Table.")

                            elif component_name == "Analysis Results":
                                st.info("Placeholder: Specific analysis results would be shown here.")

                            elif component_name == "Charts":
                                st.info("Placeholder: Charts based on active layer data would be shown.")
                                # Example: Plot histogram if active layer is suitable
                                # active_layer_name = st.session_state.active_layer
                                # if active_layer_name and active_layer_name in st.session_state.layers:
                                #      layer_data = st.session_state.layers[active_layer_name]
                                #      if isinstance(layer_data, (gpd.GeoDataFrame, pd.DataFrame)):
                                #          numeric_cols = layer_data.select_dtypes(include=np.number).columns
                                #          if not numeric_cols.empty:
                                #               col_to_plot = numeric_cols[0]
                                #               fig = px.histogram(layer_data, x=col_to_plot, title=f"Histogram of {col_to_plot}")
                                #               st.plotly_chart(fig, use_container_width=True)
                                #          else:
                                #               st.info("Active layer has no numeric columns for chart.")


                        except Exception as e:
                            st.error(f"Error rendering component '{component_name}': {e}")
                            logging.error(f"Dashboard render error ({component_name})", exc_info=True)

            # --- Delete Dashboard Section ---
            st.markdown("---") # Separator
            if st.button(f"Delete Dashboard '{selected_dashboard_name}'", key=f"delete_dash_{selected_dashboard_name}", type="secondary"):
                if selected_dashboard_name in st.session_state.dashboards:
                     del st.session_state.dashboards[selected_dashboard_name]
                     st.success(f"Dashboard '{selected_dashboard_name}' deleted.")
                     # Reset selection if possible or rerun
                     st.rerun()
                else:
                     st.error("Dashboard not found (may have been deleted).")


    with tab2:
        st.markdown("### Create a New Dashboard")
        with st.form("new_dashboard_form"):
            new_dashboard_name = st.text_input("Dashboard Name")
            new_dashboard_type = st.selectbox("Dashboard Type/Theme", [
                "General", "Urban Planning", "Environmental", "Agriculture", "Disaster"
            ], key="new_dash_type")

            available_components = ["Map View", "Data Table", "Analysis Results", "Charts"] # Add more as implemented
            selected_components = st.multiselect("Select Components for Dashboard", available_components, key="new_dash_components")

            submitted = st.form_submit_button("Create Dashboard")
            if submitted:
                if new_dashboard_name and new_dashboard_name not in st.session_state.dashboards:
                    if selected_components:
                         st.session_state.dashboards[new_dashboard_name] = {
                             'type': new_dashboard_type,
                             'components': selected_components
                         }
                         st.success(f"Dashboard '{new_dashboard_name}' created successfully!")
                    else:
                         st.warning("Please select at least one component for the dashboard.")
                elif not new_dashboard_name:
                     st.error("Dashboard name cannot be empty.")
                else:
                     st.error(f"A dashboard named '{new_dashboard_name}' already exists.")


# -------------------------------
# Home Page Content Function
# -------------------------------
def show_home_page():
    """Displays the welcome/front page content."""
    st.markdown("""
        <div class="welcome-text">
            <h2>Unleash The Power of Location Intelligence</h2>
            <p><strong>Empowering Strategic Decisions with Advanced Geospatial Insights</strong></p>
            <p>Welcome to the KANESPACE Geospatial Analysis platform, where we harness the power of spatial data to provide actionable insights. Our expertise in raster and vector analysis enables us to support strategic decision-making across various industries.</p>
            <p>Use the sidebar navigation to:</p>
            <ul>
                <li>Visit the <strong>Data Hub</strong> to upload files, connect to databases, or access web services.</li>
                <li>Explore <strong>Spatial Analysis</strong> tools for visualization, geostatistics, network analysis, and AI/ML insights.</li>
                <li>Create and view custom <strong>Dashboards</strong> combining maps, tables, and analysis results.</li>
                <li>Utilize helpful geospatial <strong>Tools</strong> like geocoding.</li>
            </ul>
            <p>Load some data in the <strong>Data Hub</strong> to get started!</p>
        </div>
    """, unsafe_allow_html=True)
    # Optionally add an image or logo here
    # st.image("path/to/your/logo.png", width=200)


# -------------------------------
# Main Application Layout
# -------------------------------
def main():
    st.sidebar.image("https://img.icons8.com/color/96/000000/globe--v1.png", width=64) # Example icon
    st.sidebar.title("Navigation")

    # *** Add "Home" to navigation ***
    app_mode = st.sidebar.radio("Choose Module", [
        "Home", # <--- New Home option
        "Data Hub",
        "Spatial Analysis",
        "Dashboards",
        "Tools"
    ], key="main_nav")

    # --- Sidebar Quick Actions ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")

    # Layer Selection
    if st.session_state.layers:
        available_layers = list(st.session_state.layers.keys())
        # Ensure active_layer exists, otherwise default to index 0 or None
        active_layer_index = 0
        if st.session_state.active_layer and st.session_state.active_layer in available_layers:
             active_layer_index = available_layers.index(st.session_state.active_layer)
        elif available_layers: # If active layer invalid but layers exist, default to first
             st.session_state.active_layer = available_layers[0]
        else: # No layers available
             st.session_state.active_layer = None

        selected_active = st.sidebar.selectbox(
            "Active Layer",
            available_layers,
            index=active_layer_index,
            key="active_layer_selector_sidebar",
            help="Select the primary layer for analysis and details."
        )
        # Update state only if selection changes
        if selected_active != st.session_state.active_layer:
            st.session_state.active_layer = selected_active
            # Avoid rerun here unless strictly necessary for other components

        # Layer Information Expander
        with st.sidebar.expander("ℹ️ Active Layer Info"):
            active_layer_name_sidebar = st.session_state.active_layer
            if active_layer_name_sidebar and active_layer_name_sidebar in st.session_state.layers:
                layer_data = st.session_state.layers[active_layer_name_sidebar]
                st.caption(f"Layer: **{active_layer_name_sidebar}**")
                if isinstance(layer_data, gpd.GeoDataFrame):
                    gdf = layer_data
                    st.write(f"Features: `{len(gdf)}`")
                    st.write(f"CRS: `{gdf.crs}`")
                    geom_types = gdf.geom_type.unique()
                    st.write(f"Geom Type(s): `{', '.join(map(str, geom_types))}`") # Ensure types are strings
                    if st.button("Show Attributes Head", key="show_head_sidebar"):
                         # *** USE THE HELPER FUNCTION HERE ***
                         st.dataframe(display_gdf_safely(gdf.head()))
                elif isinstance(layer_data, dict) and layer_data.get('type') == 'raster':
                     st.write(f"Type: `Raster`")
                     st.write(f"Path: `{layer_data.get('path', 'N/A')}`")
                elif isinstance(layer_data, dict) and layer_data.get('type') == 'wms':
                     st.write(f"Type: `WMS`")
                     st.write(f"URL: `{layer_data.get('url', 'N/A')}`")
                     st.write(f"Layers: `{layer_data.get('layers', 'N/A')}`")
                elif isinstance(layer_data, pd.DataFrame):
                     st.write(f"Type: `Tabular Data`")
                     st.write(f"Rows: `{len(layer_data)}`")
                     if st.button("Show Table Head", key="show_table_head_sidebar"):
                         st.dataframe(layer_data.head())
                else:
                     st.write("Layer type not recognized for detailed info.")
            else:
                st.write("No active layer selected.")

        # Clear Data Button
        if st.sidebar.button("Clear All Loaded Data", key="clear_data", type="secondary"):
            st.session_state.layers = {}
            st.session_state.active_layer = None
            if 'db_tables' in st.session_state: del st.session_state.db_tables # Clear DB state too
            cleanup_temp_files() # Clean up associated temp files
            st.success("All loaded data cleared.")
            st.rerun()

    else:
        st.sidebar.info("Load data using the Data Hub to get started.")


    # --- Main Content Area Routing ---
    if app_mode == "Home": # <--- Handle Home navigation
        show_home_page()

    elif app_mode == "Data Hub":
        st.header("☁️ Data Hub: Ingest and Manage Geospatial Data")
        handle_data_integration()
        if st.session_state.layers:
            with st.expander("🔍 Data Explorer (Active Layer)", expanded=True):
                active_layer_name_hub = st.session_state.active_layer
                if active_layer_name_hub and active_layer_name_hub in st.session_state.layers:
                    layer_data = st.session_state.layers[active_layer_name_hub]
                    st.markdown(f"**Previewing:** `{active_layer_name_hub}`")
                    if isinstance(layer_data, (gpd.GeoDataFrame, pd.DataFrame)):
                        # *** USE THE HELPER FUNCTION HERE ***
                        st.dataframe(display_gdf_safely(layer_data), use_container_width=True) # Show full table safely
                    elif isinstance(layer_data, dict) and layer_data.get('type') == 'raster':
                         st.info("Raster layer selected. Use Spatial Analysis module for visualization.")
                         try:
                             # Show basic raster plot here too
                             if os.path.exists(layer_data['path']):
                                 with rasterio.open(layer_data['path']) as src:
                                     fig, ax = plt.subplots()
                                     # Read only a subset for performance if raster is large
                                     show(src.read(1, window=rasterio.windows.Window(0, 0, 1000, 1000)),
                                          transform=src.window_transform(rasterio.windows.Window(0, 0, 1000, 1000)),
                                          ax=ax, title=active_layer_name_hub, cmap='viridis')
                                     st.pyplot(fig)
                             else:
                                 st.warning(f"Raster file path not found: {layer_data.get('path', 'N/A')}")
                         except Exception as e:
                              st.error(f"Could not display raster preview: {e}")
                    elif isinstance(layer_data, dict) and layer_data.get('type') == 'wms':
                         st.info("WMS layer selected. View it in the Spatial Analysis map.")
                    else:
                        st.write("Data type not directly viewable in table format.")
                else:
                    st.info("Select an active layer from the sidebar to explore its data.")

    elif app_mode == "Spatial Analysis":
        st.header("🔬 Spatial Analysis & Visualization")
        create_interactive_map() # Map is primary view here
        st.markdown("---")
        perform_advanced_analysis() # Analysis tools below the map

    elif app_mode == "Dashboards":
        st.header("📈 Dashboards")
        manage_dashboards()

    elif app_mode == "Tools":
        st.header("🛠️ Geospatial Utilities")
        tool = st.selectbox("Select Tool", [
            "Geocoding (Address to Coordinates)",
            "Coordinate Conversion (Placeholder)",
            "Data Transformation (Placeholder)"
        ], key="tools_select")

        if tool == "Geocoding (Address to Coordinates)":
            st.markdown("**Convert Address to Coordinates**")
            try:
                geocoder = Nominatim(user_agent=f"kanespace_geocoder_{random.randint(1000,9999)}") # Unique user agent
                location_query = st.text_input("Enter address or place name", key="geocode_input")
                if st.button("Geocode", key="geocode_button"):
                    if location_query:
                        with st.spinner(f"Geocoding '{location_query}'..."):
                            try:
                                loc = geocoder.geocode(location_query, timeout=10) # Add timeout
                            except Exception as geocode_err:
                                st.error(f"Geocoding service error: {geocode_err}")
                                loc = None

                            if loc:
                                st.success(f"Location Found: **{loc.address}**")
                                st.json({'Latitude': loc.latitude, 'Longitude': loc.longitude})

                                # Option to add marker to map layers
                                if st.checkbox("Add marker as a layer?", key="geocode_add_layer"):
                                    # Sanitize name for layer key
                                    sanitized_query = ''.join(filter(str.isalnum, location_query))[:20]
                                    marker_name = f"Marker_{sanitized_query}"
                                    marker_gdf = gpd.GeoDataFrame(
                                        {'address': [loc.address], 'query': [location_query]},
                                        geometry=gpd.points_from_xy([loc.longitude], [loc.latitude]),
                                        crs="EPSG:4326"
                                    )
                                    st.session_state.layers[marker_name] = marker_gdf
                                    st.info(f"Marker added as layer: '{marker_name}'")
                                    # Optionally make the new marker active
                                    st.session_state.active_layer = marker_name
                                    st.rerun() # Rerun to update layer lists immediately


                            else:
                                st.error(f"Location '{location_query}' not found or geocoding failed.")
                    else:
                        st.warning("Please enter an address or place name.")
            except Exception as e:
                st.error(f"Geocoding tool error: {str(e)}")
                logging.error("Geocoding Tool Error", exc_info=True)

        else:
            st.info(f"Tool '{tool}' is a placeholder for future implementation.")

# -------------------------------
# Run the Application & Footer
# -------------------------------
if __name__ == "__main__":
    main()

    # Footer in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        **KANESPACE Geospatial Analysis**
        *Version 4.1*
        *© {datetime.now().year}*
        """)
    # Optional cleanup button
    # if st.sidebar.button("Clean Temp Files Manual"):
    #    cleanup_temp_files()
    #    st.sidebar.success("Temp files cleaned.")