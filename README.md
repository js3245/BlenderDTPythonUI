The current blender UI has the ability to construct a solar farm in either square or rectangle shape, additional layout needs development. 
The python file: blendersolarfarm.py is the main file. Future edition requires sorting all function to reduce the complexity of the file structure.
The code requires lastest blender software, meanwhile, the code needs to be updated based on different blender version as some shortcuts might be changed. 
Pvlib needs to be installed and imported.
In blender scriptng window, select the python script and run to engage user interface. 
The code structure for modeling is as follows:
    1. Select the Solidworks STL file and enter file location.
    2. By entering the row and column number, starting coordinates and create the solar farm.
    3. Height and angle of the panels can be adjusted individually and together by rows (entering row number range).
    4. The code is combined with pvlib to perform standard tracking based on the given time and time zone.
The summary above will create a solar farm and allow the user to freely adjust the solar farm.
The digital side of the code requires IOT sensors setup and MQTT connection. 
At this moment: 1/7/2025, this README will only include the modeling part as further summary needs development. 
