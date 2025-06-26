from qgis.core import QgsVectorLayer, QgsProject

uri = "file:///D:/VJTI/SY/Sem-IV/IGT/predicted_wildfires.csv?encoding=%s&delimiter=%s&xField=%s&yField=%s&crs=%s" % ("UTF-8",",", "longitude", "latitude","epsg:4326")

#Make a vector layer
eq_layer=QgsVectorLayer(uri, "eq-data", "delimitedtext")

#Check if layer is valid
if not eq_layer.isValid():
    print("Layer not loaded")

# Add CSV data
QgsProject.instance().addMapLayer(eq_layer)