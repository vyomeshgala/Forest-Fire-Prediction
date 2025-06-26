from qgis._core import QgsGraduatedSymbolRenderer
from qgis.core import QgsVectorLayer, QgsProject, QgsSymbol, QgsRendererRange
from qgis.PyQt import QtGui

uri = "file:///D:/VJTI/SY/Sem-IV/IGT/predicted_wildfires.csv?encoding=%s&delimiter=%s&xField=%s&yField=%s&crs=%s" % ("UTF-8",",", "longitude", "latitude","epsg:4326")

#Make a vector layer
eq_layer=QgsVectorLayer(uri, "eq-data", "delimitedtext")

#Check if layer is valid
if not eq_layer.isValid():
    print("Layer not loaded")

targetField = 'Prediction'
rangeList = []
opacity = 1

# -------- Create first symbol ------------------
# define value ranges
minVal = 0.0
maxVal = 0.0

# range label
lab = 'No wildfire'

# color (green)
rangeColor = QtGui.QColor('#00ff00')

# create symbol and set properties
symbol1 = QgsSymbol.defaultSymbol(eq_layer.geometryType())
symbol1.setColor(rangeColor)
symbol1.setOpacity(opacity)

#create range and append to rangeList
range1 = QgsRendererRange(minVal, maxVal, symbol1, lab)
rangeList.append(range1)

# ---------- Create Second Symbol ------------
# define value ranges
minVal = 1
maxVal = 1

# range label
lab = 'Wildfire Predicted'

# color (orange)
rangeColor = QtGui.QColor('#ffa500')

# create symbol and set properties
symbol2 = QgsSymbol.defaultSymbol(eq_layer.geometryType())
symbol2.setColor(rangeColor)
symbol2.setOpacity(opacity)
symbol2.setSize(4)

#create range and append to rangeList
range2 = QgsRendererRange(minVal, maxVal, symbol2, lab)
rangeList.append(range2)

# ----------------------------
# create the renderer
groupRenderer = QgsGraduatedSymbolRenderer('', rangeList)
groupRenderer.setMode(QgsGraduatedSymbolRenderer.EqualInterval)
groupRenderer.setClassAttribute(targetField)

# apply renderer to layer
eq_layer.setRenderer(groupRenderer)

# Add CSV data
QgsProject.instance().addMapLayer(eq_layer)