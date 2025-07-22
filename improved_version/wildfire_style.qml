<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" alphaBand="-1" type="singlebandpseudocolor" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <colorramp type="gradient" name="[source]">
            <prop k="color1" v="0,255,0,255"/>
            <prop k="color2" v="255,0,0,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.33;255,255,0,255:0.66;255,165,0,255"/>
          </colorramp>
          <item alpha="255" value="0" label="Low Risk (0.0)" color="#00ff00"/>
          <item alpha="255" value="0.3" label="Low-Medium (0.3)" color="#ffff00"/>
          <item alpha="255" value="0.6" label="Medium-High (0.6)" color="#ffa500"/>
          <item alpha="255" value="1" label="High Risk (1.0)" color="#ff0000"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>