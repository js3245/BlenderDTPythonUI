bl_info = {
    "name": "SolarFarmDesign",
    "author": "Juanhe Shen",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3d > Tool",
    "category": "User Panel Design",
}    


import sys
import bpy
import os
import re
import math
import random
import time
import logging
import json
import subprocess
import requests

# Path to the local user site-packages where pandas is installed
user_site_packages = os.path.expanduser("~/.local/lib/python3.11/site-packages")

# Add this path to Blender's Python sys.path
if user_site_packages not in sys.path:
    sys.path.append(user_site_packages)

import numpy as np
import pandas as pd
import pytz
import pvlib
import pickle
import seaborn as sns
import requests
from requests.auth import HTTPBasicAuth
from pvlib.location import Location
from datetime import datetime
import matplotlib.pyplot as plt
from bpy_extras.object_utils import object_data_add
from datetime import datetime
from datetime import datetime
from mathutils import Vector
from mathutils import Matrix
from math import cos, sin
from math import pi
from paho.mqtt import client as mqttClient
#from azure.identity import DefaultAzureCredential
#from azure.core.exceptions import HttpResponseError
#from azure.digitaltwins.core import DigitalTwinsClient
#from azure.iot.device import IoTHubDeviceClient, Message

print(sys.version)
print(sys.executable)
print(sys.exec_prefix)

bpy.context.scene.cursor.location = (0, 0, 0)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

############################## Start ##############################

###################### !!!!!!!!!! & "Blender\3.6\python\bin\python.exe" -m pip install #module name
###################### !!!!!!!!!! Need to install pvlib and pandas in blender python environment
###################### !!!!!!!!!! Need to install matplotlib
###################### !!!!!!!!!! Need to install seaborn

############################## Defining some basic functions to be called upon in later codes
############################## Unique functions are defined within certain classes for unique operation

def show_message_box(message="", title="Message Box", icon='INFO'):
    def draw(self, context):
        self.layout.label(text=message)
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def get_collection_items(self, context):
    # Get a list of existing collection names
    collections = bpy.data.collections
    collection_items = [("", "Select Existing Collection", "")]
    for collection in collections:
        collection_items.append((collection.name, collection.name, f"Choose the {collection.name} collection"))

    return collection_items

def find_layer_collection(parent, selected_collection):
    for layer_collection in parent.children:
        if layer_collection.collection == selected_collection:
            return layer_collection
        else:
            found = find_layer_collection(layer_collection, selected_collection)
            if found:
                return found
    return None


def import_file(context, import_path):
    if os.path.isfile(import_path):
        if import_path.lower().endswith(".stl"):
            bpy.ops.import_mesh.stl(filepath=import_path)
        elif import_path.lower().endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=import_path)
    elif os.path.isdir(import_path):
        files = os.listdir(import_path)
        for file in files:
            file_path = os.path.join(import_path, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(".stl"):
                    bpy.ops.import_mesh.stl(filepath=file_path)
                elif file.lower().endswith(".fbx"):
                    bpy.ops.import_scene.fbx(filepath=file_path)


def delete_collection_recursive(collection):
    # Delete all objects in the collection
    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects:
        obj.select_set(True)
    bpy.ops.object.delete()

    # Recursively delete child collections
    for child_collection in collection.children:
        delete_collection_recursive(child_collection)

    # Remove the collection
    bpy.data.collections.remove(collection)


def delete_object_and_children(obj):
    # Create a list to hold all objects to be deleted
    objects_to_delete = []

    # Recursive function to collect all descendants of the object
    def collect_children(current_obj):
        objects_to_delete.append(current_obj)
        for child in current_obj.children:
            collect_children(child)

    # Start collecting from the given object
    collect_children(obj)

    # Deselect all objects to avoid unintended deletions
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects collected for deletion
    for del_obj in objects_to_delete:
        del_obj.select_set(True)

    # Delete all selected objects
    bpy.ops.object.delete()



def calculate_bounding_box(obj):
    if obj is not None and obj.type == 'MESH' and obj.data is not None:
        # Get the object's world matrix
        vertices = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        # Find the maximum Z coordinate value
        max_z = max(vertex.z for vertex in vertices)
        # Filter vertices that have the max Z coordinate value
        top_vertices = [vertex for vertex in vertices if vertex.z == max_z]
        # Sort the vertices by y, then by x coordinate (assumes the object is aligned with the world axes)
        top_vertices.sort(key=lambda v: (v.y, v.x))
        # Correct the order of vertices to: bottom-left, bottom-right, top-right, top-left
        if len(top_vertices) == 4:
            # Assuming top_vertices[0] is bottom-left and top_vertices[1] is bottom-right
            top_vertices = [top_vertices[0], top_vertices[1], top_vertices[3], top_vertices[2]]
        return top_vertices

    return None


def set_active_layer_collection(collection):
        root_layer_collection = bpy.context.view_layer.layer_collection
        found_collection = find_layer_collection(root_layer_collection, collection)
        
        if found_collection:
            bpy.context.view_layer.active_layer_collection = found_collection
    
    
def adjust_height(ordered_objects, panel_height):
    target_height = panel_height
    # Get the objects in panel collection
    objects_in_collection = ordered_objects
    
    if objects_in_collection:
        panel_object = objects_in_collection[1]
        rod_object = objects_in_collection[2]
        beam_object = objects_in_collection[3]
        plane_object = objects_in_collection[4]
    
        if rod_object and rod_object.type == 'MESH':
            # Calculate current object height
            current_height = rod_object.dimensions[2]
            
            # Calculate the scale factor to reach the target height
            scale_factor = target_height / current_height
            # Store the original location of the rod
            rod_original_location = rod_object.location.copy()
            
            # Apply scaling to achieve the desired height
            rod_object.scale[2] *= scale_factor
            # Calculate the difference in height
            height_difference = target_height - rod_object.dimensions[2]
            
            # Adjust the object's location based on the height difference
            rod_object.location.z = rod_original_location.z + height_difference / 2.0
            
            panel_original_location = panel_object.location.copy()
            beam_original_location = beam_object.location.copy()
            plane_original_location = plane_object.location.copy()
            
            panel_object.location.z = panel_original_location.z + height_difference 
            beam_object.location.z = beam_original_location.z + height_difference
            plane_object.location.z = plane_original_location.z + height_difference
            
        else:
            show_message_box(f"No rod object found", "def adjust_height", "ERROR")
    else:
        show_message_box(f"No object in collection", "def adjust_height", "ERROR") 
        return {'CANCELLED'}
    pass
    
    
def adjust_angle(ordered_objects, panel_collection, tilt_angle, z_rotate_angle):
    x_rotate = math.radians(tilt_angle)
    z_rotate = math.radians(z_rotate_angle)
    # Get the objects in panel collection
    objects_in_collection = ordered_objects
    
    if objects_in_collection:
        base_object = ordered_objects[0]
        panel_object = ordered_objects[1]
        beam_object = ordered_objects[3]
        plane_object = ordered_objects[4]
        
        bpy.ops.object.select_all(action='DESELECT')
        
        panel_object.select_set(True) 
        
        if beam_object:
            beam_center_location = beam_object.matrix_world.translation
                
            beam_x = beam_center_location.x
            beam_y = beam_center_location.y
            beam_z = beam_center_location.z
            rod_center_z = beam_z + 0.035
            
            bpy.context.scene.cursor.location = (beam_x, beam_y, rod_center_z)
#                bpy.context.view_layer.objects.active = panel_object
            objects_to_set_origin = [panel_object, plane_object]
            for obj in objects_to_set_origin:
                if obj and bpy.context.scene.cursor.location:
                    bpy.context.view_layer.objects.active = obj
                    # Select the object
                    obj.select_set(True)
                    bpy.ops.object.origin_set(type = 'ORIGIN_CURSOR')
                    obj.select_set(False)
                else:
                    show_message_box(f"No object in objects to set origin", "def adjust_angle", "ERROR")
                    
            panel_object.rotation_euler.x = x_rotate
            plane_object.rotation_euler.x = 0
            plane_object.rotation_euler.x = x_rotate
            base_object.rotation_euler.z = z_rotate
            
        else:
            show_message_box(f"No beam object found", "def adjust_angle", "ERROR")  
    else:
        show_message_box(f"Objects not in collection", "def adjust_angle", "ERROR") 
        
        return {'CANCELLED'}

    pass


def get_time_zone_items(self, context):
    # List of some example timezones
    time_zones = [
        ('America/New_York', "New York (EST)", "Eastern Standard Time"),
        ('America/Chicago', "Chicago (CST)", "Central Standard Time"),
        ('America/Los_Angeles', "Los Angeles (PST)", "Pacific Standard Time"),
        ('Europe/London', "London (GMT)", "Greenwich Mean Time"),
        ('Europe/Paris', "Paris (CET)", "Central European Time"),
        ('Asia/Shanghai', "Shanghai (CST)", "China Standard Time"),
        ('Asia/Tokyo', "Tokyo (JST)", "Japan Standard Time"),
        # Add more as needed
    ]
    return time_zones



def solar_angle(year, phi,L_loc,L_st, num_int):
    # Determine if it's a leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_year = 366
    else:
        days_in_year = 365
    # phi is latitude*pi/180, L_loc is longitude, L_st is the standard longitude, num_int is the time step. 
    dt = 1/num_int
    ntime = int(24/dt)
    n = np.arange(1,days_in_year+1) # nth day of the year
    B = (n-1)*360/days_in_year*pi/180 # [rad] 
    E = 229.2*(0.000075+0.001868*np.cos(B)-0.032077*np.sin(B)-0.014615*np.cos(2*B)-0.04089*np.sin(2*B))
    T_st = np.zeros((ntime, days_in_year))
    T_solar = np.zeros((ntime, days_in_year))
    for i in range(ntime):
        for j in range(days_in_year):
            T_st[i, j] = i / num_int  # Standard time
            T_solar[i, j] = T_st[i, j] - (4 * (L_st - L_loc) - E[j]) / 60  # Solar time
            
    omega = (15*T_solar-180)*pi/180 # [rad] Hour angle
    delta = (0.006918-0.399912*np.cos(B)+0.070257*np.sin(B)-0.006758*np.cos(2*B)+0.000907*np.sin(2*B)-0.002697*np.cos(3*B)+0.00148*np.sin(3*B))
    # [rad] Declination
    
    # use for loop to calculate theta_z
    theta_z = np.zeros((ntime,days_in_year))
    for i in range(ntime):
        for j in range(days_in_year):
            theta_z[i,j] = np.arccos(np.cos(phi)*np.cos(delta[j])*np.cos(omega[i,j])+np.sin(phi)*np.sin(delta[j]))
    # [rad] Zenith angle
    
    gamma_s = np.zeros((ntime, days_in_year))
    for i in range(ntime):
        for j in range(days_in_year):
            gamma_s[i,j] = np.sign(omega[i,j])*np.abs(np.arccos((np.cos(theta_z[i,j])*np.sin(phi)-np.sin(delta[j]))/(np.sin(theta_z[i,j])*np.cos(phi))))
    # [rad] Solar azimuth angle
    
    # [rad] Solar azimuth angle
    alpha_s = (np.pi/2-theta_z) # Solar altitude angle
    
    for j in range(days_in_year):
        for i in range(ntime):
            if omega[i, j] < 0:
                theta_z[i,j] = theta_z[i, j]
            elif omega[i, j] > 0:
                theta_z[i, j] = -theta_z[i, j]
                
    for j in range(days_in_year):
        for i in range(ntime):
            if alpha_s[i, j] < 0:
                theta_z[i, j] = 0  # Set zenith angle to zero
                gamma_s[i, j] = 0  # Set azimuth angle to zero
    
    return theta_z, gamma_s


def sensor_material_render(objects, material_name="Sensors Material", base_color=(1.0, 0.085, 0.0, 1.0), roughness=0.35):
        # Check if the material already exists
        material = bpy.data.materials.get(material_name)

        # If the material does not exist, create it
        if not material:
            material = bpy.data.materials.new(name=material_name)
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                # Set plastic-like properties
                bsdf.inputs['Base Color'].default_value = base_color  # Orange color
                bsdf.inputs['Metallic'].default_value = 0.0  # Non-metallic
                bsdf.inputs['Roughness'].default_value = roughness  # Somewhat glossy
            else:
                # Handle the case where the expected node isn't found
                self.report({'WARNING'}, "No Principled BSDF node found in the material.")
        
        # Assign the material to each object
        for obj in objects:
            # Check if the object is a mesh and can have material
            if obj.type == 'MESH':
                # If the object doesn't have any materials, add the new material
                if not obj.data.materials:
                    obj.data.materials.append(material)
                else:
                    # Replace the existing material with the new one
                    obj.data.materials[0] = material
                    # Optional: Report that the object already has a material
                    self.report({'INFO'}, f"Material on '{obj.name}' updated to {material_name}.")


def power_generation_result_plot(file_path, save_image_name, current_blend_directory):
    power_df = pd.read_csv(file_path)
    power_df["DateTime"] = pd.to_datetime(power_df["DateTime"])
    power_df["Hour"] = power_df["DateTime"].dt.hour
    
    # Summing the power generation and converting to MWh
    total_power_MWh = power_df['p_mp'].sum() / 1000000  # Convert from Wh to MWh
    plt.figure(figsize=(11, 6))
    
    plt.plot(power_df['Hour'], power_df['p_mp'], linewidth=4, color='red')
    plt.title("Power Generation Over 1 Day", fontsize=20, weight='bold', color = 'red')
    plt.xlabel("Hour", fontsize=15, weight='bold', color='red')
    plt.ylabel("Power Generation (Wh)", fontsize=15, weight='bold', color='red')
    
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3) 
  
    ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=6)

    # Adjust the axis labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)  
        label.set_weight('bold')  
        label.set_color('purple')
    
    # Annotate total power generation in MWh on the graph
    plt.figtext(0.15, 0.85, f'Total Power: {total_power_MWh:.2f} MWh', fontsize=15, color='green', weight='bold')
    image_path = os.path.join(current_blend_directory, save_image_name)
    
    # Save the plot to an image file in the directory of the Blender file
    plt.savefig(image_path)
    plt.close()
    
    return image_path


def start_end_tracking_hour_read(file_path):
    result_df = pd.read_csv(file_path)
    starttrackhour = result_df["start_tracking_hour"]
    endtrackhour = result_df["end_tracking_hour"]

    return starttrackhour, endtrackhour


def enhance_material_with_emission(obj, emission_strength=2.0):
    # Get the material
    mat = obj.data.materials[0]

    # Use nodes in the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs['Strength'].default_value = emission_strength

    # Ensure there's an image texture node and it's connected
    texture_node = None
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            texture_node = node
            break

    if not texture_node:
        # If there isn't an image texture node, we should create one
        # This would require loading the image as a texture again
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.image = bpy.data.images['NameOfYourImage']  # Use the correct image name
    
    # Connect the texture node to the emission node
    links = mat.node_tree.links
    links.new(texture_node.outputs['Color'], emission_node.inputs['Color'])

    # Link emission to the material output
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break

    if output_node:
        links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    # Make sure the material's blend mode is set to OPAQUE
    mat.blend_method = 'OPAQUE' 
    
    
def pkl_file_plotread(file_path, save_image_name, current_hour, current_blend_directory):
    day_index = 0  # Need to change into current_day_of_year if pkl file is modified
    hour_index = current_hour-1
    # Define the color bar range
    colorbar_min_value = 0  # Minimum value for the color bar
    colorbar_max_value = 2000*3600/1000000 # Maximum value for the color bar
    
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Changing the unit to mol/m^2/hr
    field_data_sim = data[day_index, hour_index, :, :] * 3600 / 1000000
    field_data_sim = field_data_sim.T
    
    plt.figure(figsize=(15, 11))  # Adjusted figure size
    heatmap = sns.heatmap(field_data_sim, cmap="plasma", cbar=True, vmin=colorbar_min_value, vmax=colorbar_max_value)
    plt.title('Radiation Heatmap for Hour: {}'.format(current_hour), fontsize = 30, fontweight='bold')
    
    # Get the original axis ticks
    xticks = [int(label) for label in heatmap.get_xticks()]
    yticks = [int(label) for label in heatmap.get_yticks()]
    
    plt.xlabel('Latitude (Length:m)', fontsize = 30, fontweight='bold') 
    plt.ylabel('Longitude (Width:m)', fontsize = 30, fontweight='bold')  

    plt.xticks(xticks, [str(x / 10) for x in xticks])
    plt.yticks(yticks, [str(y / 10) for y in yticks])
    # Add a label to the color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Radiation (mol/m²/hr)', fontsize=30, fontweight='bold')  # Label units
    # Customizing the color bar tick labels
    colorbar.ax.tick_params(labelsize=20, labelcolor='black', width=3)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontsize(15)
        label.set_fontweight('bold')
    plt.tight_layout()  # Adjust layout to utilize space
#    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    # Adjust the vertical line to reflect the transposition
#    plt.axhline(y=54, color='b', linestyle='--')  # Change from axvline to axhline
    image_path = os.path.join(current_blend_directory, save_image_name)
    plt.savefig(image_path)
    plt.close()
    
    return image_path
    
    
        
############################## User Defined Input for properties
bpy.types.Scene.collection_selection = bpy.props.EnumProperty(
    name="Collection Selection",
    description="Select an existing collection or create a new one",
    items=get_collection_items,
)

bpy.types.Scene.collection_name = bpy.props.StringProperty(
    name="Collection Name",
    default="MyNewCollection",
    description="Name of the new collection"
)

bpy.types.Scene.import_path = bpy.props.StringProperty(
    name="File Path",
    description="Select an STL or FBX file or a folder containing STL/FBX files",
    subtype='FILE_PATH',
)

# Define panel amount per row, and rows per solar farm
bpy.types.Scene.starting_x_cor = bpy.props.IntProperty(
    name="Starting x location",
    default = 0,
)
bpy.types.Scene.starting_y_cor = bpy.props.IntProperty(
    name="Starting y location",
    default = 0,
)
bpy.types.Scene.panels_in_one_row = bpy.props.IntProperty(
    name="Panels in 1 Row",
    default=4,
    min=1,
)
bpy.types.Scene.number_of_rows = bpy.props.IntProperty(
    name="Number of Rows",
    default=4,
    min=1,
)
bpy.types.Scene.gap_distance = bpy.props.FloatProperty(
    name="Column Spacing Distance",
    default=5.16,
    min=1,
)
bpy.types.Scene.pitch_distance = bpy.props.FloatProperty(
    name="Pitch Spacing Distance",
    default=8,
    min=5,
    step=10,  # Controls how much the value changes with each increment/decrement
    precision=1  # Number of decimal places to display
)
bpy.types.Scene.height_adjust = bpy.props.FloatProperty(
    name="Adjust Panel Height",
    default=1.6,
    min=1.6,
    step=10,
    precision=1,
)
bpy.types.Scene.tilt_angle = bpy.props.IntProperty(
    name="Tilt Angle",
    default=0,
    min=-60,
    max=60,
)
bpy.types.Scene.z_rotate_angle = bpy.props.IntProperty(
    name="Z Rotate Angle",
    default=0,
    min=0,
    max=360,
)
bpy.types.Scene.starting_row = bpy.props.IntProperty(
    name="Start Panel Row",
    default=1,
    min=1,
)
bpy.types.Scene.ending_row = bpy.props.IntProperty(
    name="End Panel Row",
    default=1,
    min=1,
)
bpy.types.Scene.latitude = bpy.props.FloatProperty(
    name="Enter Loc Latitude",
    default=42.45,
    step=1,  
    precision=2,       
) 
bpy.types.Scene.longitude = bpy.props.FloatProperty(
    name="Enter Loc Longitude",
    default=-72.62,
    step=1,  
    precision=2,     
) 
bpy.types.Scene.std_longitude = bpy.props.FloatProperty(
    name="Enter Standard Longitude",
    default=-75,
    step=1,  
    precision=2,     
)
bpy.types.Scene.timezone_select = bpy.props.EnumProperty(
    name = "Select Time Zone",
    description = "Choose your timezone",
    items = get_time_zone_items
)
bpy.types.Scene.sensorbox_id = bpy.props.StringProperty(
    name = "Sensor Box ID",
    default = "",
    description = "Enter Sensor box ID"
)
bpy.types.Scene.dli_under = bpy.props.FloatProperty(
    name = "Under Panel DLI",
    default = 20,
    step=10,  
    precision=1  
)
bpy.types.Scene.dli_between = bpy.props.FloatProperty(
    name = "Between Panel DLI",
    default = 30,
    step=10, 
    precision=1  
)
bpy.types.Scene.year_select = bpy.props.IntProperty(
    name = "Simualtion year",
    default = 2017,
    max = datetime.now().year
)
bpy.types.Scene.month_select = bpy.props.IntProperty(
    name = "Simualtion month",
    default = 5,
    min = 1,
    max = 12
)
bpy.types.Scene.day_select = bpy.props.IntProperty(
    name = "Simualtion day",
    default = 1,
    min = 1,
    max = 31
)



##### IMPORTANT: class RESULT_OT_ShowResultOperator and class External_Sim_Operator requires file path input, right now is manual input, need to change that so that it looks for the desired file and get the path. 
############################## Start ##############################

#####################
# This is the solar panel material from the CGgami-Solar-Panel-Shader.blend blender file, need to input this for rendering

current_blend_directory = os.path.dirname(bpy.data.filepath)
# Name of the .blend file with the material
material_blend_file_name = "CGgami-Solar-Panel-Shader.blend"
# Construct the full path to the .blend file with the material
material_blend_file_path = os.path.join(current_blend_directory, material_blend_file_name)
material_name = "CGgami - Solar Panel"
# Check if the material already exists in the current file
if material_name not in bpy.data.materials:
    # Material doesn't exist, so proceed with the append operation
    directory_path = material_blend_file_path + "\\Material\\"
    full_path_to_material = directory_path + material_name
    bpy.ops.wm.append(
        filepath=full_path_to_material,
        directory=directory_path,
        filename=material_name
    )
    show_message_box(f"Material '{material_name}' appended successfully.", "Material Append", 'CHECKMARK')
else:
    # Material already exists, no need to append
    show_message_box(f"Material '{material_name}' already exists. Skipping append.", "Material Append", 'INFO')


# This is connecting the IoT sensors to the system, needs to install paho 1.6.1
# Terminal: pip install paho-mqtt==1.6.1
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")  # Indicates successful connection
        # Perform actions upon successful connection
    else:
        print(f"Failed to connect, return code: {rc}")

def on_message(client, userdata, message):
    payload_str = message.payload.decode('utf-8')  # Convert bytes to string
    payload_data = json.loads(payload_str)  # Parse JSON string to Python dictionary
    
    device_id = payload_data.get("end_device_ids", {}).get("device_id")
    decoded_payload = payload_data.get("uplink_message", {}).get("decoded_payload", {})
    
    t_e = decoded_payload.get("t")
    Par_e = decoded_payload.get("Par")
#    Sm_e= decoded_payload.get("Sm")
#    rh_e = decoded_payload.get("rh")
    Vbat_e = decoded_payload.get("Vbat")
    
    message_display_console = f"Temperature:{t_e:.2f}, Radiation:{Par_e: 2f}, Battery Voltage:{Vbat_e: 2f}"
    message_display_twin = (
    f"Device ID {device_id}\n"
    f"Temperature: {t_e:.2f}\n"
    f"Radiation: {Par_e:.2f}\n"
#    f"Soil Moisture: {Sm_e:.2f}\n"
#    f"Relative Humidity: {rh_e:.2f}\n"
    f"Battery Voltage: {Vbat_e:.2f}"
    )
    print("")
    update_text_obj(device_id, message_display_twin, message_display_console)


def update_text_obj(device_id, message_display_twin, message_display_console):
    try:
        # Attempt to access the 'DeviceID' collection
        device_id_collection = bpy.data.collections['DeviceID']
    except KeyError:
        # This block executes if the 'DeviceID' collection does not exist
        print(f"Error: 'DeviceID' collection not found in the current Blender file.")
        return  # Exit the function early since further processing cannot be done without the collection
    
    # Iterate through objects in 'Labels' collection
    for text_object in device_id_collection.objects:
        # Check if the text object name matches the expected format
        if text_object.name == f"{device_id}_label":
            # Update the matched text object with the new message
            text_object.data.body = message_display_twin
            return  # Exit the function after updating the text object
    # If no text object was found for the device ID
    print(f"No twin found for device ID {device_id}. Message: {message_display_console}")
    

def connect_to_mqtt():
    broker_address = "nam1.cloud.thethings.network"
    broker_port = 1883
    topic = "v3/solar-site@ttn/devices/+/up"
    user = "solar-site@ttn"
    password = "NNSXS.WTMES2Y2VJDZ24VA5GXJQKHSQKKYTEJ5OSDTR3Y.2ET4ISRBGSY7WBI3CD22UYBS4WYCLDH4K3YSC6EFXD6GHYCEQLNA"

    client = mqttClient.Client("Python")
    client.username_pw_set(user, password=password)
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(broker_address, broker_port, 60)
    client.subscribe(topic)

    # Start a separate thread to handle MQTT communication
    client.loop_start()

# Call the function to connect to MQTT
#connect_to_mqtt()    



#def calculate_model_center(active_object):
#    if active_object and active_object.type == 'MESH':
#        # Get the bounding box of the active object
#        bounding_box = active_object.bound_box

#        # Calculate the center of the bounding box
#        center = sum((Vector(b) for b in bounding_box), Vector()) / 8

#        return center
#    else:
#        return Vector((0, 0, 0))  # Return the origin if no valid active object



############################## Class and Operator Section, Operation Starts ##############################
########## Not needed at this moment!!!!!!!!!!!!!!!!!!!!!
########## File Import Class ##########
#class P_PT_File_Import (bpy.types.Panel):
#    bl_label = "File Import"
#    bl_idname = "P_PT_File_Import"
#    bl_space_type = 'VIEW_3D'
#    bl_region_type = 'UI'
#    bl_category = 'FarmDesign'
#    bl_options = {'DEFAULT_CLOSED'}
#    
#    def draw(self, context):
#        layout = self.layout


#### Child Class: Collection Active
#class P_PT_Collection_Activation (bpy.types.Panel):
#    bl_label = "Set Active Workspace"
#    bl_idname = "PANEL_PT_Collection_Activation"
#    bl_space_type = 'VIEW_3D'
#    bl_region_type = 'UI'
#    bl_category = 'FarmDesign'
#    bl_parent_id = "P_PT_File_Import"  # Specify the parent panel's bl_idname
#    bl_options = {'DEFAULT_CLOSED'}
#    
#    def draw(self, context):
#        layout = self.layout
#        

#        row = layout.row()
#        row.prop(context.scene, "collection_selection", text="")
#        row = layout.row()
#        row.operator("solar_farm.set_active_collection", text="Set Workspace")


#class OT_ActiveCollectionOperator(bpy.types.Operator):
#    bl_idname = "solar_farm.set_active_collection"
#    bl_label = "Set Active Collection"

#    def execute(self, context):
#        selected_collection_name = context.scene.collection_selection
#        selected_collection = bpy.data.collections.get(selected_collection_name)
#        
#        if selected_collection:
#            # Set the desired collection as the active collection
#            root_layer_collection = bpy.context.view_layer.layer_collection
#            found_collection = find_layer_collection(root_layer_collection, selected_collection)

#            if found_collection:
#                bpy.context.view_layer.active_layer_collection = found_collection

#        return {'FINISHED'}
#        
#        
#### Child Class: Collection Create
#class P_PT_Collection_Creation (bpy.types.Panel):
#    bl_label = "Collections Creator"
#    bl_idname = "PANEL_PT_Collection_Creation"
#    bl_space_type = 'VIEW_3D'
#    bl_region_type = 'UI'
#    bl_category = 'FarmDesign'
#    bl_parent_id = "P_PT_File_Import"  # Specify the parent panel's bl_idname
#    bl_options = {'DEFAULT_CLOSED'}
#    
#    def draw(self, context):
#        layout = self.layout
#        
#        row = layout.row()
#        row.label(text="New Collection")
#        row = layout.row()
#        row.prop(context.scene, "collection_name", text="", icon = 'COLLECTION_NEW')
#        row = layout.row()
#        row.operator("solar_farm.create_collection", text="Create Collection")
#        


#class OT_CreateCollectionOperator(bpy.types.Operator):
#    bl_idname = "solar_farm.create_collection"
#    bl_label = "Create New Collection"

#    def execute(self, context):
#        # Access the collection_name property from the scene
#        new_collection_name = context.scene.collection_name
#        
#        # Create a new collection with the user-specified name
#        new_collection = bpy.data.collections.new(name=new_collection_name)
#        
#        # Access the currently selected collection
#        selected_collection = context.collection  # Modify this line accordingly
#        
#        # Link the new collection to the selected collection
#        selected_collection.children.link(new_collection)
#        
#        return {'FINISHED'}        
#    

#### Child Class: File Import
#class P_PT_Model_Import (bpy.types.Panel):
#    bl_label = "Model File Selector"
#    bl_idname = "P_PT_Model_Import"
#    bl_space_type = 'VIEW_3D'
#    bl_region_type = 'UI'
#    bl_category = 'FarmDesign'
#    bl_parent_id = "P_PT_File_Import"  # Specify the parent panel's bl_idname
#    bl_options = {'DEFAULT_CLOSED'}
#    
#    def draw(self, context):
#        layout = self.layout
#        row = layout.row()
#        row.label(text="Model File Selection", icon = 'FILEBROWSER')
#        row = layout.row()
#        row.prop(context.scene, "import_path", text='')
#        row = layout.row()
#        row.operator("solar_farm.import_file", text="Import File")


#### Operator that Import Files
#class OT_ImportFileOperator(bpy.types.Operator):
#    bl_idname = "solar_farm.import_file"
#    bl_label = "Import File"

#    def execute(self, context):
#        import_path = bpy.path.abspath(context.scene.import_path)
#        if import_path:
#            try:
#                import_file(context, import_path)
#            except Exception as e:
#                self.report({'ERROR'}, f"Error importing file: {str(e)}")
#                return {'CANCELLED'}
#                
#        return {'FINISHED'}



##########  Multiple Solar Panel Creation Class  ##########
class P_PT_Site_Creation (bpy.types.Panel):
    bl_label = "Solar Site Creation"
    bl_idname = "P_PT_Site_Creation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout


### Child Class: Column and Row Creation
class P_PT_Column_Row_Setup (bpy.types.Panel):
    bl_label = "Multiple Panel Creation"
    bl_idname = "P_PT_Column_Row_Setup"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Site_Creation"  # Specify the parent panel's bl_idname
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Entering starting x location
        row = layout.row()
        row.prop(context.scene, "starting_x_cor", text="X coordinate")
        
        row = layout.row()
        row.prop(context.scene, "starting_y_cor", text="Y coordinate")
        
        # Create a row for specifying panels in one row
        row = layout.row()
        row.prop(context.scene, "panels_in_one_row", text="Panels in One Row")

        # Create a row for specifying the number of rows
        row = layout.row()
        row.prop(context.scene, "number_of_rows", text="Number of Rows")
        
        row = layout.row()
        row.prop(context.scene, "gap_distance", text="Column Gap")
        
        row = layout.row()
        row.prop(context.scene, "pitch_distance", text="Pitch Distance")
        
        row = layout.row()
        row.prop(context.scene, "import_path", text='')
    
        row = layout.row()
        row.operator("solar_site.farm_generator", text = "Create Site")
        
        row = layout.row()
        row.operator("object.delete_solar_site", text = "Delete Site (Everything)")


class OT_SolarSiteCreator (bpy.types.Operator):
    bl_idname = "solar_site.farm_generator"
    bl_label = "Generate Solar Site"
    
    def execute(self, context):
        # User input used as properties input
        number_columns = context.scene.panels_in_one_row
        number_rows = context.scene.number_of_rows
        
        self.delete_excess_rows(number_rows)
        
        file_path = bpy.path.abspath(context.scene.import_path)

        for row in range(1, number_rows + 1):
            # Create a parent collection for each row
            row_collection_name = f"Row{row}"
            row_collection = bpy.data.collections.get(row_collection_name)
            if row_collection:
                self.delete_excess_columns(row_collection, number_columns)
            if not row_collection:
                # If row collections do not exist then create within the selected collection
                row_collection = bpy.data.collections.new(row_collection_name)
                bpy.context.scene.collection.children.link(row_collection)

                self.set_active_layer_collection(row_collection)
                
            for col in range(1, number_columns + 1):
                create_collection_name = f"Panel{row}-{col}"

                # Check if the panel collection already exists
                panel_collection = bpy.data.collections.get(create_collection_name)

                if not panel_collection:
                    # If the panel collection doesn't exist, create it
                    panel_collection = bpy.data.collections.new(create_collection_name)  
                    row_collection.children.link(panel_collection)    
    
                else:
                    # Print a message if the collection already exists
                    show_message_box(f"Collection already exists. Checking Files", "OT_SolarSiteCreator", "INFO")
                
                self.set_active_layer_collection(panel_collection)
                # Import the model files into the panel collection
                self.import_model_parts(file_path, panel_collection, context)             
                
                order_mapping0 = {
                    "SolarPanelAssem-Base-1": 0,
                    "SolarPanelAssem-PanelArray101P-1": 1,
                    "SolarPanelAssem-Rod-1": 2,
                    "SolarPanelAssem-RotateBeam-1": 3
                }
                # Get the objects in the panel collection
                objects_in_collection = panel_collection.objects
                
                if objects_in_collection:
                    # Sort objects based on order_mapping
                    ordered_objects0 = sorted(objects_in_collection, key=lambda obj: order_mapping0.get(self.get_base_name(obj.name), float('inf')))
                panel_object = ordered_objects0[1]
                panel_top_surface = calculate_bounding_box(panel_object)    
                
                z_margin = 0.007
                # Adjust Z coordinate by subtracting the margin
                adjusted_panel_top_surface = [Vector((v.x, v.y, v.z - z_margin)) for v in panel_top_surface]                                
                self.panel_material_render(ordered_objects0, material_name="Panel Material", metallic=1.0, roughness=0.0)
                
                self.render_plane_create(context, adjusted_panel_top_surface, panel_collection)      
                
                order_mapping = {
                    "SolarPanelAssem-Base-1": 0,
                    "SolarPanelAssem-PanelArray101P-1": 1,
                    "SolarPanelAssem-Rod-1": 2,
                    "SolarPanelAssem-RotateBeam-1": 3,
                    "RenderPlane": 4
                }
                
                # Get the objects in the panel collection
                objects_in_collection = panel_collection.objects
                
                if objects_in_collection:
                    # Sort objects based on order_mapping
                    ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
                # After import, move the panel according to their collection
                self.adjust_location(order_mapping, ordered_objects, row, col, context)
                
#        show_message_box(f"Generated {number_rows * number_columns} collections.", "OT_SolarSiteCreator", "CHECKMARK")
        bpy.ops.object.select_all(action = 'DESELECT')
        
        return {'FINISHED'}


    def delete_excess_rows(self, intended_rows):
        prefix = "Row"
        for collection in bpy.data.collections:
            if collection.name.startswith(prefix):
                row_number = int(collection.name[len(prefix):])
                if row_number > intended_rows:
                    delete_collection_recursive(collection)


    def delete_excess_columns(self, row_collection, intended_columns):
        prefix = "Panel"
        for panel_collection in row_collection.children:
            if panel_collection.name.startswith(prefix):
                parts = panel_collection.name.split('-')
                if len(parts) > 1:
                    column_number = int(parts[1])
                    if column_number > intended_columns:
                        delete_collection_recursive(panel_collection)


    def set_active_layer_collection(self, collection):
        root_layer_collection = bpy.context.view_layer.layer_collection
        found_collection = find_layer_collection(root_layer_collection, collection)
        
        if found_collection:
            bpy.context.view_layer.active_layer_collection = found_collection


    def import_model_parts(self, file_path, panel_collection, context):
        # Define a dictionary to map object names to their desired order
        base_part = ["SolarPanelAssem-Base-1", "SolarPanelAssem-PanelArray101P-1", "SolarPanelAssem-Rod-1", "SolarPanelAssem-RotateBeam-1"]
        existing_objects = []

        for base_name in base_part:
            # Check for the base name only, ignoring the suffix
            regex_pattern = re.escape(base_name) + r"(?:\.\d+)?"
            existing_objects.extend([obj for obj in panel_collection.objects if re.match(regex_pattern, obj.name)])

        if len(existing_objects) == len(base_part):
            self.report({'INFO'}, f"Parts already exist in collection '{panel_collection.name}'. Skipping import.")

        else:
            import_file(context, file_path)
            self.report({'INFO'}, "Import Completed, Move to Next Collection") 
#            bpy.ops.object.select_all(action = 'DESELECT')         
    
        
    def render_plane_create(self, context, vertices, collection): 
        base_plane = ["RenderPlane"]
        existing_planes = []
        
        for base_name in base_plane:
            # Check for the base name only, ignoring the suffix
            regex_pattern = re.escape(base_name) + r"(?:\.\d+)?"
            existing_planes.extend([obj for obj in collection.objects if re.match(regex_pattern, obj.name)])
            
        if len(existing_planes) == len(base_plane):
            self.report({'INFO'}, f"Plane already exists in collection '{collection.name}'. Skipping creation.")
            
        else:
            mesh = bpy.data.meshes.new(name="RenderPlane")
            obj = bpy.data.objects.new("RenderPlane", mesh)
            
            # Link the object to the scene's collection
            context.collection.objects.link(obj)
            
            # Set the active object to our new object and select it
            context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Create the mesh from given vertices, edges, and faces
            mesh.from_pydata(
                vertices,  # Vertices
                [],  # Edges
                [(0, 1, 2, 3)]  # Faces
            )
            mesh.update()
            
            # Create UV map
            uv_layer = mesh.uv_layers.new()
            uv_layer.data[0].uv = (0, 0)
            uv_layer.data[1].uv = (1, 0)
            uv_layer.data[2].uv = (1, 1)
            uv_layer.data[3].uv = (0, 1)

            # Get the material
            material = bpy.data.materials.get("CGgami - Solar Panel")
            
            if material:
                # Assign the material to the plane
                if len(obj.data.materials):
                    obj.data.materials[0] = material  # Replace existing material
                else:
                    obj.data.materials.append(material)  # Add new material
                    
                material.use_nodes = True
                nodes = material.node_tree.nodes
                
                # Adjust the desired values (These names should match the names in the node tree)
                if 'Size' in nodes:  # Check if the 'Size' node exists
                    nodes['Size'].outputs['Value'].default_value = 13.5
                if 'Busbars' in nodes:  # Check if the 'Busbars' node exists
                    nodes['Busbars'].outputs['Value'].default_value = 2.5
                if 'Cell Fingers' in nodes:  # Check if the 'Cell Fingers' node exists
                    nodes['Cell Fingers'].outputs['Value'].default_value = 10
                    
            else:
                # If the material doesn't exist, show an error message
                self.show_message_box(f"Material '{material_name}' not found. Please make sure it's created and named correctly.", "Material Error", "ERROR")
                    
    
    
    def panel_material_render(self, objects, material_name="Panel Material", metallic=1.0, roughness=0.2):
        # Check if the material already exists
        material = bpy.data.materials.get(material_name)

        # If the material does not exist, create it
        if not material:
            material = bpy.data.materials.new(name=material_name)
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Metallic'].default_value = metallic
                bsdf.inputs['Roughness'].default_value = roughness
            else:
                self.report({'WARNING'}, "No Principled BSDF node found in the material.")
        
        # Assign the material to each object
        for obj in objects:
            # Check if the object is a mesh and can have material
            if obj.type == 'MESH':
                # If the object doesn't have any materials, add the new material
                if not obj.data.materials:
                    obj.data.materials.append(material)
                else:
                    # Optional: Report that the object already has a material
                    self.report({'INFO'}, f"Object '{obj.name}' already has a material assigned.")
        
    
    def adjust_location(self, order_mapping, objects_in_collection, row, col, context):   
        column_gap = context.scene.gap_distance
        pitch_gap = context.scene.pitch_distance
        x_dis = context.scene.starting_x_cor
        y_dis = context.scene.starting_y_cor
        
        # Set the location based on row and col
        x_location = x_dis + (col - 1) * column_gap  # Adjust this value based on your desired spacing
        y_location = y_dis + (row - 1) * pitch_gap
        z_location = 0
        
        
        if objects_in_collection:
            # Sort objects based on order_mapping
            ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))

            # Set the parent object based on order_mapping
            parent_object = ordered_objects[0]

            # Apply parent-child relationship to objects in the collection
            bpy.ops.object.select_all(action='DESELECT')
            for obj in ordered_objects:
                obj.select_set(True)

            if parent_object.children:
                # If the parent object already has children, set its location
                parent_object.location = (x_location, y_location, z_location)
            else:
                # If the parent object doesn't have children, set it as the parent object
                bpy.context.view_layer.objects.active = parent_object
                bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

                # Set the origin of the parent object to the geometry's bounding box center
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

                # Set the parent object's location
                parent_object.location = (x_location, y_location, z_location)

    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]
                

class OT_DeleteSiteOperator(bpy.types.Operator):
    bl_idname = "object.delete_solar_site"
    bl_label = "Delete Solar Site"
    
    def execute(self, context):
#    # Define the prefix used during the creation of collections and objects
#        prefix = "Row"
#    
#        # Iterate through collections and objects to delete those with the defined prefix
#        for collection in bpy.data.collections:
#            if collection.name.startswith(prefix):
#                delete_collection_recursive(collection)

#        show_message_box(f"Deleted Solar Site collections and objects.", "OT_DeleteSiteOperator", "CHECKMARK")
#        
#        return {'FINISHED'}

        # Will delete everything in the scene
        root_collections = list(bpy.context.scene.collection.children)
        
        # Iterate through the root collections and delete them recursively
        for collection in root_collections:
            delete_collection_recursive(collection)

        # Optionally, clear all objects in the scene as well
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        show_message_box("Deleted all collections and objects in the scene.", "OT_DeleteSiteOperator", "CHECKMARK")
        
        return {'FINISHED'}

    

########## Panel Customization Class ##########

class P_PT_Panel_Edition (bpy.types.Panel):
    bl_label = "Single Panel Modification"
    bl_idname = "P_PT_Panel_Edition"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
    


### Child Class: Panel Movement
class P_PT_Panel_Move (bpy.types.Panel):
    bl_label = "Panel Position Adjust"
    bl_idname = "P_PT_Panel_Move"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Panel_Edition"  # Specify the parent panel's bl_idname
    bl_options = {'DEFAULT_CLOSED'}
    
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text="Location Adjust", icon='EMPTY_ARROWS')
        row = layout.row()
        row.operator("object.custom_location", text="Set Custom Location")
        

class OT_LocationOperator(bpy.types.Operator):
    bl_idname = "object.custom_location"
    bl_label = "Change Location"

    x_loc: bpy.props.FloatProperty(name="X Location")
    y_loc: bpy.props.FloatProperty(name="Y Location")
    z_loc: bpy.props.FloatProperty(name="Z Location")
    
    initial_location = None  # Store the initial location of the object
    
    def execute(self, context):
        # Find the collection of the active object
        active_obj = context.active_object
        
        order_mapping = {
                "SolarPanelAssem-Base-1": 0,
                "SolarPanelAssem-PanelArray101P-1": 1,
                "SolarPanelAssem-Rod-1": 2,
                "SolarPanelAssem-RotateBeam-1": 3,
                "RenderPlane": 4
            }
        selected_collection = None
        
        if active_obj:
            if self.initial_location is None:
                self.initial_location = active_obj.location.copy()  # Store the initial location
            
            selected_collection = None
            
            for collection in bpy.data.collections:
                if active_obj.name in collection.objects:
                    selected_collection = collection
                    break
            
            if selected_collection:
                # Apply parent-child relationship to objects in the collection
                bpy.ops.object.select_all(action='DESELECT')
                for obj in selected_collection.objects:
                    obj.select_set(True)
                
                ordered_objects = sorted(selected_collection.objects, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
                # Set the first object in the collection as active
                object_base = ordered_objects[0]
                object_panel = ordered_objects[1]
                object_rod = ordered_objects[2]
                object_beam = ordered_objects[3]
                object_plane = ordered_objects[4]
                
                # Set the first object in the collection as active
                first_object = object_base

                if first_object.children:  # Check if the first object is already a parent
                    # Set the first object as the active object
                    bpy.context.view_layer.objects.active = first_object
                    # Set the first object in the collection as the location reference
                    first_object.location = (self.x_loc, self.y_loc, self.z_loc)
                else:
                    # Set the first object as the parent object
                    bpy.context.view_layer.objects.active = first_object
                    bpy.ops.object.parent_set(type='OBJECT')
                    
                    # Set the origin of the parent object to the geometry's bounding box center
                    context.view_layer.objects.active = first_object
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                    
                    # Set the first object in the collection as the location reference
                    first_object.location = (self.x_loc, self.y_loc, self.z_loc)
        
        return {'FINISHED'}
    
    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]

    # Find the active object
    def invoke(self, context, event):
        active_obj = context.active_object
        
        selected_collection = None
        for collection in bpy.data.collections:
            if active_obj.name in collection.objects:
                selected_collection = collection
                break
                
        if selected_collection:
            first_obj = selected_collection.objects[0]
            
            
        # Set the initial location values based on the active object's location
        self.x_loc = first_obj.location.x
        self.y_loc = first_obj.location.y
        self.z_loc = first_obj.location.z
            
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "x_loc")
        layout.prop(self, "y_loc")
        layout.prop(self, "z_loc")




### Child Class: Height Adjustment
class P_PT_Panel_Height (bpy.types.Panel):
    bl_label = "Panel Height Adjust"
    bl_idname = "P_PT_Panel_Height"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Panel_Edition"  # Specify the parent panel's bl_idname
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text="Adjust Panel Height:", icon='CON_SAMEVOL')
        row = layout.row()
        row.prop(context.scene, "height_adjust", text="Panel Height")
        row = layout.row()
        row.operator("object.custom_height", text='Adjust Panel Height')
        
        
class OT_PanelHeightOperator (bpy.types.Operator):
    bl_idname = "object.custom_height"
    bl_label = "Change Height"
    
    def execute(self, context):
        panel_height = context.scene.height_adjust   
        order_mapping = {
            "SolarPanelAssem-Base-1": 0,
            "SolarPanelAssem-PanelArray101P-1": 1,
            "SolarPanelAssem-Rod-1": 2,
            "SolarPanelAssem-RotateBeam-1": 3,
            "RenderPlane": 4
            }
        selected_collection = None
        active_obj = context.active_object
        
        for collection in bpy.data.collections:
            if active_obj.name in collection.objects:
                selected_collection = collection
                break
        
        if selected_collection:
            objects_in_collection = selected_collection.objects
            
            ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
            adjust_height(ordered_objects, panel_height)
            
        else:
            show_message_box(f"No collection found", "OT_PanelHeightOperator", "ERROR")   
            
        return {'FINISHED'}
        
        
    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]    



### Child Class: Panel Rotation
class P_PT_Panel_Rotation (bpy.types.Panel):
    bl_label = "Panel Angle Adjust"
    bl_idname = "P_PT_Panel_Angle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Panel_Edition"  # Specify the parent panel's bl_idname
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text="Adjust Single Panel Angle:", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        row = layout.row()
        row.prop(context.scene, "tilt_angle", text="Tilt Angle")
        row = layout.row()
        row.prop(context.scene, "z_rotate_angle", text="Z Rotate Angle")
        row = layout.row()
        row.operator("object.panel_rotate", text='Set Panel Angle')



class OT_PanelAngleOperator (bpy.types.Operator):
    bl_idname = "object.panel_rotate"
    bl_label = "Panel Rotate"
    
    def execute(self, context):
        tilt_angle = context.scene.tilt_angle
        z_rotate_angle = context.scene.z_rotate_angle
        order_mapping = {
            "SolarPanelAssem-Base-1": 0,
            "SolarPanelAssem-PanelArray101P-1": 1,
            "SolarPanelAssem-Rod-1": 2,
            "SolarPanelAssem-RotateBeam-1": 3,
            "RenderPlane": 4
        }
        selected_collection = None
        # Find the collection of the active object
        active_obj = context.active_object
       
        for collection in bpy.data.collections:
            if active_obj.name in collection.objects:
                selected_collection = collection
                break
            
        if selected_collection:
            objects_in_collection = selected_collection.objects
            
            ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
            adjust_angle(ordered_objects, selected_collection, tilt_angle, z_rotate_angle)
            
        else:
            show_message_box(f"No collection found", "OT_PanelAngleOperator", "ERROR")   
            
        return {'FINISHED'}
            
    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]
            
    
    
########## Row Modification ##########
class P_PT_Row_Mod(bpy.types.Panel):
    bl_label = "Solar Site Adjustments"
    bl_idname = "P_PT_Row_Mod"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        
#### Child Class: Row Angle Adjust
class P_PT_Row_AngleMod(bpy.types.Panel):
    bl_label = "Panel Properties/Orientation Adjust"
    bl_idname = "P_PT_Row_AngleMod"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Row_Mod"  # Specify the parent panel's bl_idname
    bl_options = {'DEFAULT_CLOSED'}       
    
    def draw(self, context):
        layout = self.layout
        
        # Entering the starting row
        row = layout.row()
        row.prop(context.scene, "starting_row", text="Starting Row")

        # Entering the ending row
        row = layout.row()
        row.prop(context.scene, "ending_row", text="Ending Row")
        
        row = layout.row()
        row.prop(context.scene, "height_adjust", text="Panel Height")
    
        row = layout.row()
        row.prop(context.scene, "tilt_angle", text="Tilt Angle")
        
        row = layout.row()
        row.prop(context.scene, "z_rotate_angle", text="Z Rotate Angle")
        
        row = layout.row()
        row.prop(context.scene, "latitude", text="Latitude")
        
        row = layout.row()
        row.prop(context.scene, "longitude", text = "Longitude")
        
        row = layout.row()
        row.prop(context.scene, "std_longitude", text = "STD Longitude")
        
        row = layout.row()
        row.label(text="Select Timezone")

        # Create another row for the actual dropdown property
        row = layout.row()
        row.prop(context.scene, "timezone_select", text="")
           
        row = layout.row()
        row.operator("row_panel.adjust", text = "Custom Modify Preview")
        
        row = layout.row()
        row.operator("row_panel.standard_track", text = "Standard Tracking")
        
        row = layout.row()
        row.operator("row_panel.custom_track", text = "Custom Tracking(Simulation Required)")


class OT_PanelRowModOperator(bpy.types.Operator):
    bl_idname = "row_panel.adjust"
    bl_label = "Row Panel Height/Angle Set"
    
    def execute(self, context):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        # User input used as properties input
        start_row = context.scene.starting_row
        end_row = context.scene.ending_row
        panel_height = context.scene.height_adjust
        tilt_angle = context.scene.tilt_angle
        z_rotate_angle = context.scene.z_rotate_angle
        order_mapping = {
            "SolarPanelAssem-Base-1": 0,
            "SolarPanelAssem-PanelArray101P-1": 1,
            "SolarPanelAssem-Rod-1": 2,
            "SolarPanelAssem-RotateBeam-1": 3,
            "RenderPlane": 4
        }
        
        bpy.context.view_layer.objects.active = None
        
        if start_row > end_row:
            show_message_box(f"Error in row value entering.", "OT_PanelRowModOperator", "ERROR")
            
        else:
            for row_num in range(start_row, end_row+1):
                row_collection_name = f"Row{row_num}"
                row_collection = bpy.data.collections.get(row_collection_name)
                
                if row_collection:
                    # iterate through panel collections in the row collection
                    for panel_collection in row_collection.children:
                        if panel_collection:
                            objects_in_collection = panel_collection.objects
                            ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
                            
                            set_active_layer_collection(panel_collection)
                            adjust_height(ordered_objects, panel_height)
                            adjust_angle(ordered_objects, panel_collection, tilt_angle, z_rotate_angle)
                           
                        else: 
                            show_message_box(f"No panel collection in current row", "OT_PanelRowModOperator", "ERROR")
                        
                else:
                    show_message_box(f"No row found in scene. Check if solar site exist.", "OT_PanelRowModOperator", "ERROR")
        
        return {'FINISHED'}
    
    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]


    
# the center from mid line 35.77mm


class OT_PanelStandardTrackOperator(bpy.types.Operator):
    bl_idname = 'row_panel.standard_track'
    bl_label = 'Standard Track Panel Adjust'
    
    def execute(self, context):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        # User input used as properties input
        start_row = context.scene.starting_row
        end_row = context.scene.ending_row
        panel_height = context.scene.height_adjust
        z_rotate_angle = context.scene.z_rotate_angle
        num_int = 6  # Number of time setup, if 30mins, num_int = 2
        latitude = (context.scene.latitude)*pi/180
        longitude = context.scene.longitude
        L_st = context.scene.std_longitude
        tz = context.scene.timezone_select
        timezone = pytz.timezone(tz)
        current_time = datetime.now(timezone)
        current_hour = current_time.hour
        current_year = current_time.year
        current_day_of_year = current_time.timetuple().tm_yday

        ########## Art Model solar angle calculation
        theta_z, gamma_s = solar_angle(current_year, latitude, longitude, L_st, num_int)
        sun_angle_degrees = math.degrees(theta_z[current_hour, current_day_of_year - 1]) 

        ########## Pvlib solar angle calculation
        site_location = Location(latitude, longitude, tz)
        # Generate a date range starting from the current time
        solar_position = site_location.get_solarposition(current_time)
        
        zenith_angle = solar_position['zenith'].iloc[0]
        azimuth_angle = solar_position['azimuth'].iloc[0]
        
        ########## Determine panel rotation based on azimuth, angle adjust
        if azimuth_angle > 180:
            # Sun is in the east
            rotation_direction = 'EAST'
            tilt_angle = min(zenith_angle, 60)
            z_rotate_angle = context.scene.z_rotate_angle
        else:
            # Sun is in the west
            rotation_direction = 'WEST'
            tilt_angle = max(-zenith_angle, -60)
            z_rotate_angle = context.scene.z_rotate_angle

#        tilt_angle = sun_angle_degrees
#        z_rotate_angle = context.scene.z_rotate_angle
        
        order_mapping = {
            "SolarPanelAssem-Base-1": 0,
            "SolarPanelAssem-PanelArray101P-1": 1,
            "SolarPanelAssem-Rod-1": 2,
            "SolarPanelAssem-RotateBeam-1": 3,
            "RenderPlane": 4
        }
        
        bpy.context.view_layer.objects.active = None
        
        if start_row > end_row:
            show_message_box(f"Error in row value entering.", "OT_PanelRowModOperator", "ERROR")
            
        else:
            for row_num in range(start_row, end_row+1):
                row_collection_name = f"Row{row_num}"
                row_collection = bpy.data.collections.get(row_collection_name)
                
                if row_collection:
                    # iterate through panel collections in the row collection
                    for panel_collection in row_collection.children:
                        if panel_collection:
                            objects_in_collection = panel_collection.objects
                            ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
                            
                            set_active_layer_collection(panel_collection)
                            adjust_height(ordered_objects, panel_height)
                            adjust_angle(ordered_objects, panel_collection, tilt_angle, z_rotate_angle)
                           
                        else: 
                            show_message_box(f"No panel collection in current row", "OT_PanelRowModOperator", "ERROR")
                        
                else:
                    show_message_box(f"No row found in scene. Check if solar site exist.", "OT_PanelRowModOperator", "ERROR")
        
        return {'FINISHED'}
    
    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]




class OT_PanelCustomTrackOperator(bpy.types.Operator):
    bl_idname = 'row_panel.custom_track'
    bl_label = 'Custom Track Panel Adjust'
    
    def execute(self, context):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        # User input used as properties input
        start_row = context.scene.starting_row
        end_row = context.scene.ending_row
        panel_height = context.scene.height_adjust
        tilt_vertical = 90
        z_rotate_angle = context.scene.z_rotate_angle
        num_int = 1  # Number of time setup, if 30mins, num_int = 2
        latitude = (context.scene.latitude)*pi/180
        longitude = context.scene.longitude
        L_st = context.scene.std_longitude
        tz = context.scene.timezone_select
        timezone = pytz.timezone(tz)
        current_time = datetime.now(timezone)
        current_hour = current_time.hour
        current_year = current_time.year
        current_day_of_year = current_time.timetuple().tm_yday
        W_p = 4  # As you mentioned W_p is a constant
        year = context.scene.year_select  # Constant year
        W_r = context.scene.pitch_distance - 4  # Assuming row_width maps to W_r
        H = context.scene.height_adjust  # Assuming panel_height maps to H

        results_dir = os.path.join(current_blend_directory, "solar-farm-design-blender", "examples", "results")
        
        # Construct simulation directory name
        sim_name = f'sat_ew_bi_wr{W_r}_wp{W_p}_h{H}_yr{year}'
        sim_results_dir = os.path.join(results_dir, sim_name)

        # Construct file path for the power generation results
        daily_result_filename = f"daily_results_{year}.csv"
        daily_result_path = os.path.join(sim_results_dir, daily_result_filename)
        
        # Check if the file exists
        if os.path.exists(daily_result_path):
            starttrackhr, endtrackhr = start_end_tracking_hour_read(daily_result_path)
            starttrackhr = int(starttrackhr.iloc[0])  # Assuming the result is a Series; adjust accordingly.
            endtrackhr = int(endtrackhr.iloc[0])
            self.report({'INFO'}, f"Result CSV read, start and end tracking hour loaded")
        else:
            self.report({'ERROR'}, f"Check if simulation is completed")
            return {"CANCELLED"}

        ########## Art Model solar angle calculation
        theta_z, gamma_s = solar_angle(current_year, latitude, longitude, L_st, num_int)
        sun_angle_degrees = math.degrees(theta_z[current_hour, current_day_of_year - 1]) 

        ########## Pvlib solar angle calculation
        site_location = Location(latitude, longitude, tz)
        # Generate a date range starting from the current time
        solar_position = site_location.get_solarposition(current_time)
        
        zenith_angle = solar_position['zenith'].iloc[0]
        azimuth_angle = solar_position['azimuth'].iloc[0]
        
        ########## Determine panel rotation based on azimuth, angle adjust
        if current_hour >= starttrackhr and current_hour < endtrackhr:
            if azimuth_angle < 180:
                # Sun is in the east
                rotation_direction = 'EAST'
                tilt_angle = min(zenith_angle, 60)
                z_rotate_angle = context.scene.z_rotate_angle
            else:
                # Sun is in the west
                rotation_direction = 'WEST'
                tilt_angle = max(-zenith_angle, -60)
                z_rotate_angle = context.scene.z_rotate_angle
        else: 
            if azimuth_angle > 180:
                # Sun is in the east
                rotation_direction = 'EAST'
                tilt_angle = max(-zenith_angle, -60)
                z_rotate_angle = context.scene.z_rotate_angle
            else:
                # Sun is in the west
                rotation_direction = 'WEST'
                tilt_angle = min(zenith_angle, 60)
                z_rotate_angle = context.scene.z_rotate_angle
        
#        ########## This section will remove the restriction on tilit angle 60 to -60 degree, for showing purpose
#        if current_hour >= starttrackhr and current_hour < endtrackhr:
#            if azimuth_angle < 180:
#                # Sun is in the east
#                rotation_direction = 'EAST'
#                tilt_angle = min(zenith_angle, 70)
#                z_rotate_angle = context.scene.z_rotate_angle
#            else:
#                # Sun is in the west
#                rotation_direction = 'WEST'
#                tilt_angle = max(-zenith_angle, -70)
#                z_rotate_angle = context.scene.z_rotate_angle
#        else: 
#            if azimuth_angle > 180:
#                # Sun is in the east
#                rotation_direction = 'EAST'
#                tilt_angle = -(tilt_vertical - zenith_angle)
#                z_rotate_angle = context.scene.z_rotate_angle
#            else:
#                # Sun is in the west
#                ### Angle is adjusted to match the graph
#                rotation_direction = 'WEST'
#                tilt_angle = tilt_vertical - (zenith_angle-25)
#                z_rotate_angle = context.scene.z_rotate_angle

#        tilt_angle = sun_angle_degrees
#        z_rotate_angle = context.scene.z_rotate_angle
        
        order_mapping = {
            "SolarPanelAssem-Base-1": 0,
            "SolarPanelAssem-PanelArray101P-1": 1,
            "SolarPanelAssem-Rod-1": 2,
            "SolarPanelAssem-RotateBeam-1": 3,
            "RenderPlane": 4
        }
        
        bpy.context.view_layer.objects.active = None
        
        if start_row > end_row:
            show_message_box(f"Error in row value entering.", "OT_PanelRowModOperator", "ERROR")
            
        else:
            for row_num in range(start_row, end_row+1):
                row_collection_name = f"Row{row_num}"
                row_collection = bpy.data.collections.get(row_collection_name)
                
                if row_collection:
                    # iterate through panel collections in the row collection
                    for panel_collection in row_collection.children:
                        if panel_collection:
                            objects_in_collection = panel_collection.objects
                            ordered_objects = sorted(objects_in_collection, key=lambda obj: order_mapping.get(self.get_base_name(obj.name), float('inf')))
                            
                            set_active_layer_collection(panel_collection)
                            adjust_height(ordered_objects, panel_height)
                            adjust_angle(ordered_objects, panel_collection, tilt_angle, z_rotate_angle)  
                        else: 
                            show_message_box(f"No panel collection in current row", "OT_PanelRowModOperator", "ERROR")
                        
                else:
                    show_message_box(f"No row found in scene. Check if solar site exist.", "OT_PanelRowModOperator", "ERROR")
        
        return {'FINISHED'}
    
    def get_base_name(self, full_name):
        # Extract the base name by removing the appended numbers
        return full_name.split('.')[0]


#### Digital Twin Section:
# In the beginning of the code, There is a function defined to extract digital twin data from the ADT website every 5 seconds

########## Digital Twiin Main panel ##########
class P_PT_Digital_Twin (bpy.types.Panel):
    bl_label = "Digital Twin Info Explorer"
    bl_idname = "P_PT_Digital_Twin"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        
### Child Class: Create Text for DT
class P_PT_Twin_Create(bpy.types.Panel):
    bl_label = "DT Creator"
    bl_idname = "P_PT_Text_Create"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Digital_Twin"  # Specify the parent panel's bl_idname
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        layout.label(text = "Enter Sensor Box ID")
        
        row = layout.row()
        layout.prop(context.scene, "sensorbox_id", text = "", icon = "GHOST_ENABLED")
        
        row = layout.row()
        layout.operator("twin.create", text="Create Twin")
        
        row = layout.row()
        layout.operator("twin.display", text="Show/Hide Twin")
        
        ROW = layout.row()
        layout.operator("twin.delete", text="Delete Twin")
        
#        row = layout.row()
#        layout.operator("twin.dataresult", text="Result Data Request")



# Define an operator to create twin
class OT_CreateTwinOperator(bpy.types.Operator):
    bl_idname = "twin.create"
    bl_label = "Create Twin"

    def execute(self, context):
        sensor_id = context.scene.sensorbox_id.strip().lower()
        
        if not sensor_id:
            show_message_box(f"Sensor Box Value Error", "OT_CreateTwinOperator", "ERROR")
            return {'CANCELLED'}
        
        if sensor_id in bpy.data.objects:
            show_message_box(f"Sensor Box Already Exists", "OT_CreateTwinOperator", "INFO")
            return {'CANCELLED'}
        
        sensor_file_name = "SensorBox.STL"
        sensor_model_path = os.path.join(current_blend_directory, sensor_file_name)
        
        if "DeviceID" not in bpy.data.collections:
            result_collection = bpy.data.collections.new("DeviceID")
            bpy.context.scene.collection.children.link(result_collection)
        else:
            result_collection = bpy.data.collections["DeviceID"]
        
        bpy.ops.object.select_all(action='DESELECT')
        
        if sensor_model_path:
            try:
                import_file(context, sensor_model_path)
            except Exception as e:
                self.report({'ERROR'}, F"Error importing file")
                return{'CANCELLED'}
        
        
        # Move the imported object to the "SimResult" collection
        imported_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
        for obj in imported_objs:
            # Make sure the object is not linked to any other collection
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            # Link the object to the "SimResult" collection
            result_collection.objects.link(obj)
            obj.name = sensor_id
            obj.location = (8, -5, 0)  # Set to your desired location
            obj.rotation_euler = (math.radians(0), math.radians(0), math.radians(0))
            sensor_material_render([obj])
        
        active_obj = context.active_object
        
        selected_collection = None
        for collection in bpy.data.collections:
            if active_obj.name in collection.objects:
                selected_collection = collection
                break
        
        if selected_collection:
            root_layer_collection = bpy.context.view_layer.layer_collection
            found_collection = find_layer_collection(root_layer_collection, selected_collection)

            if found_collection:
                bpy.context.view_layer.active_layer_collection = found_collection
                
            text_name = active_obj.name + "_label"
            existing_text = bpy.data.objects.get(text_name)
            
            if existing_text:
                print("Twin already exists")
                
            else:
                box_location = active_obj.matrix_world.translation
                bpy.ops.object.text_add(location=(0, 0, box_location.z+12))
                obj = bpy.context.object
                obj.name = text_name
                obj.rotation_euler = (math.radians(90), math.radians(0), 0)
                
                # Set color for the text
                obj.data.materials.append(bpy.data.materials.new(name="TextMaterial"))
                obj.data.materials[0].diffuse_color = (random.random(), random.random(), random.random(), 1)
                obj.hide_viewport = True
                obj.hide_render = True
                
                # Set text size
                obj.data.size = 0.5  # Adjust the size as needed
                # Set the extrude parameter to give the text depth
                obj.data.extrude = 0.05  # Change the value to adjust the depth of the text
                # Parent the text object to the active object (sensor box)
                obj.parent = active_obj
                
                show_message_box(f"Twin Created", "OT_CreateTwinOperator", "CHECKMARK")
        else:
            show_message_box(f"Collection Not Found", "OT_CreateTwinOperator", "ERROR")
        
        return {'FINISHED'}
    
class OT_DeleteTwinOperator(bpy.types.Operator):
    bl_idname = "twin.delete"
    bl_label = "Delete Selected Twin"
    
    def execute(self, context):
        active_obj = context.active_object
        
        if active_obj:
            delete_object_and_children(active_obj)
            show_message_box(f"Selected Twin Deleted", "OT_DeleteTwinOperator", "CHECKMARK")
        else:
            show_message_box(f"No Twin Selected, check again", "OT_DeleteTwinOperator", "ERROR")
            
        return {'FINISHED'}
        
    
    
class OT_ShowHideTextOperator(bpy.types.Operator):
    bl_idname = "twin.display"
    bl_label = "Show/Hide Twin"

    def execute(self, context):
        active_obj = context.active_object

        # Find the associated text object based on the active object's name
        text_name = active_obj.name + "_label"
        text_object = bpy.data.objects.get(text_name)
        
        if text_object:
            # Toggle the visibility of the associated text object
            text_object.hide_viewport = not text_object.hide_viewport
            text_object.hide_render = not text_object.hide_render

            print(f"Text visibility toggled for {active_obj.name}")
        else:
            print("Associated twin not found")

        return {'FINISHED'}
    

class OT_TwinDataObtain(bpy.types.Operator):
    bl_idname = "twin.dataresult"
    bl_label = "Obtain Twin Data (daily)"
    
    def execute(self, context):
        active_obj = context.active_object

        # Find the associated text object based on the active object's name
        deviceID = active_obj.name
        twinID = active_obj.name + "_label"
        device_object = bpy.data.objects.get(twinID)

        if device_object:
            # Setup for HTTP request
            INFLUXDB_HOST = "https://en-ma-maxzhang-node18.coecis.cornell.edu"
            USERNAME = 'mzinfluxdb'   # Enter Username
            PASSWORD = """o]'x!N)"Sar?/v5aG@up2<9g-cZ^Y8.<"""   # Enter the passcode
            login_url = f"{INFLUXDB_HOST}/login"
            
             # Start session with basic authentication
            session = requests.Session()
            session.auth = (USERNAME, PASSWORD)
            
            # First, try to log in (if there's a login endpoint)
            login_response = session.get(login_url)
            print(login_response.text)
            
            if login_response.status_code == 200:
                self.report({'INFO'}, "Login successful")
                # Proceed with the query after successful login
                DB_NAME = 'Digital_Agriculture'   # Database name
                QUERY = f"""
                SELECT mean("par") AS "mean_par"
                FROM "{DB_NAME}"."autogen"."digital_agriculture"
                WHERE time > now() - 1d AND "deviceID"='{deviceID}' GROUP BY time(1h) FILL(null)
                """
                query_url = f"{INFLUXDB_HOST}:8086/query?db={DB_NAME}&q={QUERY}"
                query_response = session.get(query_url)
                
                if query_response.status_code == 200:
                    self.report({'INFO'}, "Data retrieval successful.")
                    print(query_response.text)
                    return {'FINISHED'}
                
                else:
                    self.report({'ERROR'}, f"Failed to retrieve data: {query_response.status_code} - {query_response.text}")
                return {'CANCELLED'}
            
            else:
                self.report({'ERROR'}, f"Failed to log in: {login_response.status_code} - {login_response.text}")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Current object has no twin association")
            
        return {'FINISHED'}
                



### Child Class: Adjust Location
class P_PT_BoxLocation_Adjust(bpy.types.Panel):
    bl_label = "Sensor Location Adjust"
    bl_idname = "P_PT_Location_Adjust"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_Digital_Twin"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        active_object = context.active_object

        # Add draggable sliders to adjust location
        row = layout.row()
        row.prop(active_object, "location", index=0, text="X")
        row.prop(active_object, "location", index=1, text="Y")



### Operator that select all object in the selected collection and create a parent relationship
   
                    
#        model_collection = bpy.data.collections.get(selected_collection_name)
#        
#        if model_collection:
#            # Create an empty list to store world space bounding box coordinates
#            bounding_box = []

#            # Iterate through all objects in the collection and calculate their world space bounding box coordinates
#            for obj in model_collection.objects:
#                for v in obj.bound_box:
#                    # Assuming 'v' represents a coordinate, perform matrix multiplication properly
#                    world_coord = obj.matrix_world @ Vector(v[:])  # Make sure 'v' is a coordinate-like iterable
#                    bounding_box.append(world_coord)

#            # Calculate the min and max coordinates of the bounding box
#            min_coords = [min(coord[i] for coord in bounding_box) for i in range(3)]
#            max_coords = [max(coord[i] for coord in bounding_box) for i in range(3)]

#            # Print the calculated bounding box coordinates
#            print("Min Coords:", min_coords)
#            print("Max Coords:", max_coords)
    
    

########## External Simulation Connect Class ##########
class P_PT_External_Sim (bpy.types.Panel):
    bl_label = "External Simulation"
    bl_idname = "P_PT_External_Sim"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        
### Child Class: Radiation Agrivoltaic Sim
class P_PT_RadAgriSim (bpy.types.Panel):
    bl_label = "Radiation Agrivoltaic Sim"
    bl_idname = "P_PT_RadAgriSim"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SolarFarmDesign'
    bl_parent_id = "P_PT_External_Sim"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
#        row = layout.row()
#        row.prop(context.scene, "latitude", text="Latitude")
#        row = layout.row()
#        row.prop(context.scene, "longitude", text="Longitude")
        
#        row = layout.row()
#        row.prop(context.scene, "height_adjust", text="Panel Height")
#    
#        row = layout.row()
#        row.prop(context.scene, "tilt_angle", text="Tilt Angle")
        
#        row = layout.row()
#        row.()

        row = layout.row()
        row.prop(context.scene, "year_select", text= "Year")
        row.prop(context.scene, "month_select", text= "Month")
        row.prop(context.scene, "day_select", text = "Day")

#        row = layout.row()
#        row.prop(context.scene, "dli_under", text="Desired DLI Under Panel")
        
        row = layout.row()
        row.prop(context.scene, "dli_between", text="DLI Target")
        
        row = layout.row()
        row.operator("sim.run", text = "Run Simulation")
        
        row = layout.row()
        row.operator("show.radiationresult", text = 'Show Radiation Result')
        
        row = layout.row()
        row.operator("show.powerresult", text = 'Show Power Result')




class OT_ExternalSimOperator(bpy.types.Operator):
    bl_idname = "sim.run"
    bl_label = "External Sim Run"

    def execute(self, context):
        year = context.scene.year_select
        month = context.scene.month_select
        day = context.scene.day_select
        selected_date = datetime(year, month, day)
        selected_day_of_year = selected_date.timetuple().tm_yday
#        print(selected_day_of_year)
        user_inputs = {
            'start_year': year,
            'num_array': context.scene.number_of_rows,
            'row_width': context.scene.pitch_distance - 4,
            'panel_height': context.scene.height_adjust,
            'loc_latitude': context.scene.latitude,
            'loc_longitude': context.scene.longitude,
            'desired_under_DLI_value': context.scene.dli_under,
            'desired_between_DLI_value': context.scene.dli_between -1,
            'starting_day': selected_day_of_year,
        }
        
        simulation_path = os.path.join("solar-farm-design-blender", "examples", "feed_back_control_execute.py")
        script_path = os.path.join(current_blend_directory, simulation_path)
        # Join the current directory with the relative folder name
        cwd_path = os.path.join(current_blend_directory, "solar-farm-design-blender")
        
#        script_path = r"C:\Cornell\Research\MasterProject\PPFD\solar-farm-design-blender\examples\feed_back_control_execute.py"
        
        python_interpreter = "C://Users//jhesh//anaconda3//envs/sf-design//python.exe"

        # Check if the file exists at the specified path
        if not os.path.isfile(script_path):
            self.report({'ERROR'}, f"File not found at path: {script_path}")
            return {"CANCELLED"}

        with open(script_path, "r") as file:
            script_content = file.read()
            
        for var, value in user_inputs.items():
            # Regex pattern to find the variable assignment (handles both integers and floats)
            pattern = rf"{var}\s*=\s*[\d\.]+"
            replacement = f"{var} = {value}"
            script_content = re.sub(pattern, replacement, script_content, flags=re.MULTILINE)

        with open(script_path, "w") as file:
            file.write(script_content)

        # Execute the external script
        try:
            subprocess.run([python_interpreter, script_path], check=True, cwd=cwd_path)
            show_message_box(f"External Script Run Completed", "OT_ExternalSimOperator", "CHECKMARK") 
            
        except subprocess.CalledProcessError as e:
            show_message_box(f"External Script Error", "OT_ExternalSimOperator", "ERROR")

            return {'CANCELLED'}
        
        return {'FINISHED'}
        
        
##################################### IMPORTANT #####################################
################### This operator requires activing the add on: Images as Planes, Edit-Preferences-AddOn

class OT_ShowRadResultOperator(bpy.types.Operator):
    bl_idname = "show.radiationresult"
    bl_label = "Show Simulation Result"
    
    ########### Change this into directory in the future
#    figure_path = r"C:\Cornell\Research\MasterProject\PPFD\solar-farm-design-blender\examples\figures\radiation_par_ew_sat0.0_sat_ew_bi_wr4_wp4_h1.6_yr2017.png"
#    

    def execute(self, context):
        W_p = 4  # As you mentioned W_p is a constant
        year = context.scene.year_select  # Constant year
        W_r = context.scene.pitch_distance - 4  # Assuming row_width maps to W_r
        H = context.scene.height_adjust  # Assuming panel_height maps to H
        tz = context.scene.timezone_select
        timezone = pytz.timezone(tz)
        current_time = datetime.now(timezone)
        current_hour = current_time.hour
        current_year = current_time.year
        current_day_of_year = current_time.timetuple().tm_yday
         
        results_dir = os.path.join(current_blend_directory, "solar-farm-design-blender", "examples", "results")
        
        # Construct simulation directory name
        sim_name = f'sat_ew_bi_wr{W_r}_wp{W_p}_h{H}_yr{year}'
        sim_results_dir = os.path.join(results_dir, sim_name)
        
        # Construct file path for the power generation results
        rad_result_pkl = f'radiation_t_par_array.pkl'
        rad_result_path = os.path.join(sim_results_dir, rad_result_pkl)
            
        if os.path.exists(rad_result_path):
            rad_heatmap_image_path = pkl_file_plotread(rad_result_path, "rad_heatmap_plot.png", current_hour, current_blend_directory)
            self.report({'INFO'}, f"Image saved at: {rad_heatmap_image_path}")
        else:
            self.report({'ERROR'}, f"Result file not found: {rad_result_path}")
            
        # Extract the file name and directory from figure_path
        file_name = rad_heatmap_image_path.split("\\")[-1]
        directory = rad_heatmap_image_path.rsplit("\\", 1)[0]
        
        if "RadiationSimResult" not in bpy.data.collections:
            result_collection = bpy.data.collections.new("RadiationSimResult")
            bpy.context.scene.collection.children.link(result_collection)
        else:
            result_collection = bpy.data.collections["RadiationSimResult"]
           
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        
        # Check if an object with the same name already exists
        obj_name = file_name.rsplit('.', 1)[0]  # Assuming the object name will be the file name without the extension
        if obj_name in bpy.data.objects:
            # Delete the existing object
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
        
        # Import the image as a plane
        bpy.ops.import_image.to_plane(files=[{"name": file_name}], directory=directory)
        
        # Move the imported object to the "SimResult" collection
        imported_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
        for obj in imported_objs:
            # Make sure the object is not linked to any other collection
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            # Link the object to the "SimResult" collection
            result_collection.objects.link(obj)
            # Adjust location, scale, and rotation here
            obj.location = (7.5, 9, 0)  # Set to your desired location
            obj.scale = (38, 36, 37)  # Set to your desired scale
            obj.rotation_euler = (math.radians(0), math.radians(0), math.radians(-90))
        
        return {'FINISHED'}
        

##################################### IMPORTANT #####################################
################### This operator requires activing the add on: Images as Planes, Edit-Preferences-AddOn

class OT_ShowPowResultOperator(bpy.types.Operator):
    bl_idname = "show.powerresult"
    bl_label = "Show Simulation Result"


    def execute(self, context):
        W_p = 4  # As you mentioned W_p is a constant
        year = context.scene.year_select  # Constant year
        W_r = context.scene.pitch_distance - 4  # Assuming row_width maps to W_r
        H = context.scene.height_adjust  # Assuming panel_height maps to H

        power_storage_file_name = "PowerStorage.STL"
        power_unit_model_path = os.path.join(current_blend_directory, power_storage_file_name)
        
        if "PowerSimResult" not in bpy.data.collections:
            result_collection = bpy.data.collections.new("PowerSimResult")
            bpy.context.scene.collection.children.link(result_collection)
        else:
            result_collection = bpy.data.collections["PowerSimResult"]
            
        bpy.ops.object.select_all(action = 'DESELECT')
        
        power_model_name = "PowerStorage"
        obj_name = power_model_name.rsplit('.',1)[0]
        if obj_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink = True)
            
        # Import the power storage stl file using import_file function\
        if power_unit_model_path:
            try:
                import_file(context, power_unit_model_path)
            except Exception as e:
                self.report({'ERROR'}, f"Error importing file")
                return {'CANCELLED'}
            
        # Move the imported object to the "SimResult" collection
        imported_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
        for obj in imported_objs:
            # Make sure the object is not linked to any other collection
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            # Link the object to the "SimResult" collection
            result_collection.objects.link(obj)
            obj.location = (25, 7.8, 0)  # Set to your desired location
            obj.rotation_euler = (math.radians(0), math.radians(0), math.radians(-90))
            power_material = bpy.data.materials.get("Panel Material")
            obj.data.materials.append(power_material)
            
#        power_result_path = r"C:\Cornell\Research\MasterProject\PPFD\solar-farm-design-blender\examples\results\sat_ew_bi_wr4.0_wp4_h2.0_yr2017\solar_gen_ac_sat_bi_year2017.csv"
#        power_result_image_path = power_generation_result_plot(power_result_path, "predict_power_result.png", current_blend_directory) 
#        self.report({'INFO'}, f"Image saved at: {power_result_image_path}") 
  
        
        results_dir = os.path.join(current_blend_directory, "solar-farm-design-blender", "examples", "results")
        
        # Construct simulation directory name
        sim_name = f'sat_ew_bi_wr{W_r}_wp{W_p}_h{H}_yr{year}'
        sim_results_dir = os.path.join(results_dir, sim_name)

        # Construct file path for the power generation results
        power_result_filename = f"solar_gen_ac_sat_bi_year{year}.csv"
        power_result_path = os.path.join(sim_results_dir, power_result_filename)
        
        # Check if the file exists
        if os.path.exists(power_result_path):
            power_result_image_path = power_generation_result_plot(power_result_path, "predict_power_result.png", current_blend_directory) 
            self.report({'INFO'}, f"Image saved at: {power_result_image_path}")
        else:
            self.report({'ERROR'}, f"Result file not found: {power_result_path}")
        
        active_obj = context.active_object
        power_storage_loc = active_obj.matrix_world.translation
        
        selected_collection = None
        for collection in bpy.data.collections:
            if active_obj.name in collection.objects:
                selected_collection = collection
                break
        
        if selected_collection:
            root_layer_collection = bpy.context.view_layer.layer_collection
            found_collection = find_layer_collection(root_layer_collection, selected_collection)

            if found_collection:
                bpy.context.view_layer.active_layer_collection = found_collection
        
        file_name = power_result_image_path.split("\\")[-1]
        directory = power_result_image_path.rsplit("\\",1)[0]
        image_name = file_name.rsplit('.', 1)[0]
        if image_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[image_name], do_unlink=True)
        
        bpy.ops.import_image.to_plane(files=[{"name": file_name}], directory=directory)
        
        image_plane = context.active_object
        enhance_material_with_emission(image_plane, emission_strength=1)
        image_plane.location = (power_storage_loc.x, power_storage_loc.y, power_storage_loc.z + 10)
        power_storage_obj = bpy.data.objects.get("PowerStorage")
        
        if power_storage_obj:
            image_plane.parent = power_storage_obj
            image_plane.matrix_parent_inverse = power_storage_obj.matrix_world.inverted()
            image_plane.scale = (10, 10, 10)
            image_plane.rotation_euler = (math.radians(90), 0, math.radians(270))

        # Deselect everything first
        bpy.ops.object.select_all(action='DESELECT')
        for area in bpy.context.screen.areas:  # Iterate through all areas in the current screen
            if area.type == 'VIEW_3D':  # Find the 3D Viewport
                space = area.spaces.active
                space.overlay.show_relationship_lines = False
            
        return {'FINISHED'}
          



def register():
#    bpy.utils.register_class(P_PT_File_Import)
#    
#    bpy.utils.register_class(P_PT_Collection_Activation)
#    bpy.utils.register_class(OT_ActiveCollectionOperator)
#    
#    bpy.utils.register_class(P_PT_Collection_Creation)
#    bpy.utils.register_class(OT_CreateCollectionOperator)

#    bpy.utils.register_class(P_PT_Model_Import)
#    bpy.utils.register_class(OT_ImportFileOperator)
    
    bpy.utils.register_class(P_PT_Site_Creation)
    bpy.utils.register_class(P_PT_Column_Row_Setup)
    bpy.utils.register_class(OT_SolarSiteCreator)
    bpy.utils.register_class(OT_DeleteSiteOperator)
    
    bpy.utils.register_class(P_PT_Panel_Edition)
    
    bpy.utils.register_class(P_PT_Panel_Move)
    bpy.utils.register_class(OT_LocationOperator)
    bpy.utils.register_class(P_PT_Panel_Height)
    bpy.utils.register_class(OT_PanelHeightOperator)
#    bpy.utils.register_class(RESET_TRANSFORM_OT_Operator)
    bpy.utils.register_class(P_PT_Panel_Rotation)
    bpy.utils.register_class(OT_PanelAngleOperator)
    
    bpy.utils.register_class(P_PT_Row_Mod)
    bpy.utils.register_class(P_PT_Row_AngleMod)
    bpy.utils.register_class(OT_PanelRowModOperator)
    bpy.utils.register_class(OT_PanelStandardTrackOperator)
    bpy.utils.register_class(OT_PanelCustomTrackOperator)
    
    bpy.utils.register_class(P_PT_Digital_Twin)
#    bpy.utils.register_class(P_PT_Collection_Activation_DigitalTwin)
    bpy.utils.register_class(P_PT_Twin_Create)
    bpy.utils.register_class(P_PT_BoxLocation_Adjust)
    bpy.utils.register_class(OT_CreateTwinOperator)
    bpy.utils.register_class(OT_ShowHideTextOperator)
    bpy.utils.register_class(OT_DeleteTwinOperator)
    bpy.utils.register_class(OT_TwinDataObtain)
    
    bpy.utils.register_class(P_PT_External_Sim)
    bpy.utils.register_class(P_PT_RadAgriSim)
    bpy.utils.register_class(OT_ExternalSimOperator)
    bpy.utils.register_class(OT_ShowRadResultOperator)
    bpy.utils.register_class(OT_ShowPowResultOperator)

    
def unregister():
#    bpy.utils.unregister_class(P_PT_File_Import)
#    
#    bpy.utils.unregister_class(P_PT_Collection_Activation)
#    bpy.utils.unregister_class(OT_ActiveCollectionOperator)
#    
#    bpy.utils.unregister_class(P_PT_Collection_Creation)
#    bpy.utils.unregister_class(OT_CreateCollectionOperator)
#    
#    bpy.utils.unregister_class(P_PT_Model_Import)
#    bpy.utils.unregister_class(OT_ImportFileOperator)
    
    bpy.utils.unregister_class(P_PT_Site_Creation)
    bpy.utils.unregister_class(P_PT_Column_Row_Setup)
    bpy.utils.unregister_class(OT_SolarSiteCreator)
    bpy.utils.unregister_class(OT_DeleteSiteOperator)
    
    bpy.utils.unregister_class(P_PT_Panel_Edition)
    
    bpy.utils.unregister_class(P_PT_Panel_Move)
    bpy.utils.unregister_class(OT_LocationOperator)
    bpy.utils.unregister_class(P_PT_Panel_Height)
    bpy.utils.unregister_class(OT_PanelHeightOperator)
#    bpy.utils.unregister_class(RESET_TRANSFORM_OT_Operator)
    bpy.utils.unregister_class(P_PT_Panel_Rotation)
    bpy.utils.unregister_class(OT_PanelAngleOperator)
    
    bpy.utils.unregister_class(P_PT_Row_Mod)
    bpy.utils.unregister_class(P_PT_Row_AngleMod)
    bpy.utils.unregister_class(OT_PanelRowModOperator)
    bpy.utils.unregister_class(OT_PanelStandardTrackOperator)
    bpy.utils.unregister_class(OT_PanelCustomTrackOperator)
    
    bpy.utils.unregister_class(P_PT_Digital_Twin)
#    bpy.utils.unregister_class(P_PT_Collection_Activation_DigitalTwin)
    bpy.utils.unregister_class(P_PT_Twin_Create)
    bpy.utils.unregister_class(P_PT_BoxLocation_Adjust)
    bpy.utils.unregister_class(OT_CreateTwinOperator)
    bpy.utils.unregister_class(OT_DeleteTwinOperator)
    bpy.utils.unregister_class(OT_ShowHideTextOperator)
    bpy.utils.unregister_class(OT_TwinDataObtain)
    
    bpy.utils.unregister_class(P_PT_External_Sim)
    bpy.utils.unregister_class(P_PT_RadAgriSim)
    bpy.utils.unregister_class(OT_ExternalSimOperator)
    bpy.utils.unregister_class(OT_ShowRadResultOperator)
    bpy.utils.unregister_class(OT_ShowPowResultOperator)
    
    
if __name__ == "__main__":
    register()