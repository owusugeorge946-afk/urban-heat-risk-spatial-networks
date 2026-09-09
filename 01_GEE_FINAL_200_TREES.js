// ============================================================================
// FINAL REPRODUCIBILITY SCRIPT
// Spatial Networks of Urban Surface Heat Risk:
// Connectivity, Communities and Neighbourhood Reinforcement across
// Four West African Coastal Cities
//
// Google Earth Engine — 2024 remote-sensing preprocessing and LULC
// IMPORTANT: This script uses the ORIGINAL 20 Code Editor Imports already
// present in the recovered Earth Engine project. Keep those Imports attached.
// Random Forest setting aligned with the final manuscript: 200 trees.
// ============================================================================

// --------------------------- 0. SELECT ONE CITY -------------------------------
var RUN_CITY = 'Accra'; // 'Accra', 'Lagos', 'Abidjan', or 'Freetown'

// --------------------------- 1. CITY BOUNDARIES -------------------------------
var Accra = ee.FeatureCollection('projects/ee-owusugeorge946/assets/Accra');
var Lagos = ee.FeatureCollection('projects/ee-george946/assets/lagos');
var Abidjan = ee.FeatureCollection('projects/ee-george946/assets/Abijan');
var Freetown = ee.FeatureCollection('projects/ee-george946/assets/Freetown');

// EXPECTED ORIGINAL IMPORT VARIABLE NAMES (do not rename):
// Accra: Water_Bodies_Accra, Built_Up_Accra, Vegetation_Accra,
//        Sparse_Vegetation_Accra, Bare_land_Accra
// Lagos: Water_Bodies_lagos, Built_Up_lagos, Vegetation_lagos,
//        Sparse_Vegetation_lagos, Barren_Land_lagos
// Abidjan: Water_Bodies_Abijan, Built_Up_Abijan, Vegetation_Abijan,
//          Sparse_Vegetation_Abijan, Barren_Land_Abijan
// Freetown: Water_Bodies_freetown, Built_Up_freetown, Vegetation_freetown,
//           Sparse_Vegetation_freetown, Barren_Land_freetown

// --------------------------- 2. SETTINGS --------------------------------------
var YEAR = 2024;
var RF_TREES = 200;
var RANDOM_SEED = 42;
var S2_BANDS = ['B2','B3','B4','B8','B11','B12'];
var GRID_SIZE_M = 500;
var VALID_COVERAGE_MIN = 0.50;
var EXPORT_FOLDER = 'URBAN_HEAT_NETWORK_2024';

var UTM_CRS = {
  'Accra': 'EPSG:32630',
  'Lagos': 'EPSG:32631',
  'Abidjan': 'EPSG:32630',
  'Freetown': 'EPSG:32628'
};

// --------------------------- 3. S2 MASK ---------------------------------------
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));
  return image.updateMask(mask)
    .divide(10000)
    .copyProperties(image, ['system:time_start']);
}

function makeTrainingPoints(fc, classValue) {
  return ee.FeatureCollection(fc).map(function(f) {
    return ee.Feature(f.geometry().centroid(10)).set('class', classValue);
  });
}

// --------------------------- 4. SELECT CITY + ORIGINAL TRAINING IMPORTS -------
var cityBoundary;
var cityName;
var trainingData;

if (RUN_CITY === 'Accra') {
  cityName = 'Accra';
  cityBoundary = Accra;
  trainingData = makeTrainingPoints(Water_Bodies_Accra, 0)
    .merge(makeTrainingPoints(Built_Up_Accra, 1))
    .merge(makeTrainingPoints(Vegetation_Accra, 2))
    .merge(makeTrainingPoints(Sparse_Vegetation_Accra, 3))
    .merge(makeTrainingPoints(Bare_land_Accra, 4));
}

if (RUN_CITY === 'Lagos') {
  cityName = 'Lagos';
  cityBoundary = Lagos;
  trainingData = makeTrainingPoints(Water_Bodies_lagos, 0)
    .merge(makeTrainingPoints(Built_Up_lagos, 1))
    .merge(makeTrainingPoints(Vegetation_lagos, 2))
    .merge(makeTrainingPoints(Sparse_Vegetation_lagos, 3))
    .merge(makeTrainingPoints(Barren_Land_lagos, 4));
}

if (RUN_CITY === 'Abidjan') {
  cityName = 'Abidjan';
  cityBoundary = Abidjan;
  trainingData = makeTrainingPoints(Water_Bodies_Abijan, 0)
    .merge(makeTrainingPoints(Built_Up_Abijan, 1))
    .merge(makeTrainingPoints(Vegetation_Abijan, 2))
    .merge(makeTrainingPoints(Sparse_Vegetation_Abijan, 3))
    .merge(makeTrainingPoints(Barren_Land_Abijan, 4));
}

if (RUN_CITY === 'Freetown') {
  cityName = 'Freetown';
  cityBoundary = Freetown;
  trainingData = makeTrainingPoints(Water_Bodies_freetown, 0)
    .merge(makeTrainingPoints(Built_Up_freetown, 1))
    .merge(makeTrainingPoints(Vegetation_freetown, 2))
    .merge(makeTrainingPoints(Sparse_Vegetation_freetown, 3))
    .merge(makeTrainingPoints(Barren_Land_freetown, 4));
}

if (!cityBoundary) {
  throw new Error('RUN_CITY must be Accra, Lagos, Abidjan or Freetown.');
}

var roi = cityBoundary.geometry().simplify(100);
var start = ee.Date.fromYMD(YEAR, 1, 1);
var end = start.advance(1, 'year');

print('Selected city:', cityName);
print(cityName + ' original training features:', trainingData.size());

// --------------------------- 5. SENTINEL-2 COMPOSITE --------------------------
var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2clouds);

var s2 = s2Collection.median().select(S2_BANDS).clip(roi);
print(cityName + ' Sentinel-2 scenes:', s2Collection.size());

// --------------------------- 6. SAMPLE + 70/30 SPLIT --------------------------
var samples = s2.sampleRegions({
  collection: trainingData,
  properties: ['class'],
  scale: 20,
  tileScale: 16,
  geometries: false
});

var samplesRandom = samples.randomColumn('random', RANDOM_SEED);
var trainingSamples = samplesRandom.filter(ee.Filter.lt('random', 0.70));
var validationSamples = samplesRandom.filter(ee.Filter.gte('random', 0.70));

print(cityName + ' total sampled pixels:', samples.size());
print(cityName + ' training samples:', trainingSamples.size());
print(cityName + ' validation samples:', validationSamples.size());

// --------------------------- 7. RANDOM FOREST — FINAL 200 TREES --------------
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: RF_TREES,
  bagFraction: 0.7,
  seed: RANDOM_SEED
}).train({
  features: trainingSamples,
  classProperty: 'class',
  inputProperties: S2_BANDS
});

var lulc5 = s2.classify(classifier).rename('lulc5').toByte();

// Accuracy is evaluated on the original five classes.
var validated = validationSamples.classify(classifier);
var cm = validated.errorMatrix('class', 'classification');
print(cityName + ' confusion matrix:', cm);
print(cityName + ' overall accuracy:', cm.accuracy());
print(cityName + ' Kappa:', cm.kappa());
print(cityName + ' producer accuracy:', cm.producersAccuracy());
print(cityName + ' user accuracy:', cm.consumersAccuracy());

// Final analytical classes:
// 0 water, 1 built-up, 2 vegetation (classes 2+3), 3 bare land.
var lulc4 = lulc5.expression(
  "c == 0 ? 0 : c == 1 ? 1 : (c == 2 || c == 3) ? 2 : 3",
  {c: lulc5}
).rename('lulc').toByte();

// --------------------------- 8. LANDSAT 8/9 ----------------------------------
function maskScaleLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 0).eq(0)
    .and(qa.bitwiseAnd(1 << 1).eq(0))
    .and(qa.bitwiseAnd(1 << 2).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0));

  var optical = image.select(['SR_B4','SR_B5','SR_B6'])
    .multiply(0.0000275).add(-0.2)
    .rename(['red','nir','swir1']);

  var lst = image.select('ST_B10')
    .multiply(0.00341802).add(149.0)
    .subtract(273.15).rename('lst');

  var ndvi = optical.normalizedDifference(['nir','red']).rename('ndvi');
  var ndbi = optical.normalizedDifference(['swir1','nir']).rename('ndbi');

  return ee.Image.cat([lst, ndvi, ndbi])
    .updateMask(clear)
    .updateMask(lst.gte(15).and(lst.lte(60)))
    .copyProperties(image, ['system:time_start']);
}

var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi).filterDate(start, end).map(maskScaleLandsat)
  .merge(
    ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
      .filterBounds(roi).filterDate(start, end).map(maskScaleLandsat)
  );

var thermal = landsat.median().clip(roi);
print(cityName + ' Landsat scenes:', landsat.size());

// --------------------------- 9. CITY-SPECIFIC HRI -----------------------------
function minMax(image, band) {
  return image.select(band).reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: roi,
    scale: 30,
    maxPixels: 1e13,
    tileScale: 8
  });
}

function normalize(image, band, stats) {
  var mn = ee.Number(stats.get(band + '_min'));
  var mx = ee.Number(stats.get(band + '_max'));
  return image.select(band).subtract(mn).divide(mx.subtract(mn));
}

var lstStats = minMax(thermal, 'lst');
var ndviStats = minMax(thermal, 'ndvi');
var ndbiStats = minMax(thermal, 'ndbi');

var lstN = normalize(thermal, 'lst', lstStats);
var ndviN = normalize(thermal, 'ndvi', ndviStats);
var ndbiN = normalize(thermal, 'ndbi', ndbiStats);

var hri = lstN.add(ndbiN).add(ee.Image(1).subtract(ndviN))
  .divide(3).rename('hri');

var elevation = ee.Image('USGS/SRTMGL1_003')
  .select('elevation').clip(roi);

var stack = thermal.select(['lst','ndvi','ndbi'])
  .addBands(hri)
  .addBands(elevation);

var validMask = stack.select(['lst','ndvi','ndbi','hri'])
  .mask().reduce(ee.Reducer.min());

// --------------------------- 10. 500 M GRID -----------------------------------
var proj = ee.Projection(UTM_CRS[RUN_CITY]).atScale(GRID_SIZE_M);
var metricRoi = roi.transform(UTM_CRS[RUN_CITY], 1);
var bounds = metricRoi.bounds(1);
var xy = ee.Image.pixelCoordinates(proj);

var cellId = xy.select('x').divide(GRID_SIZE_M).floor().toInt64()
  .multiply(1000000000)
  .add(xy.select('y').divide(GRID_SIZE_M).floor().toInt64())
  .rename('cell_id');

var grid = cellId.reduceToVectors({
  geometry: bounds,
  crs: proj,
  scale: GRID_SIZE_M,
  geometryType: 'polygon',
  eightConnected: false,
  labelProperty: 'cell_id',
  reducer: ee.Reducer.first(),
  maxPixels: 1e13,
  tileScale: 8
}).filterBounds(metricRoi);

grid = grid.map(function(f) {
  var clipped = f.geometry().intersection(metricRoi, 1);
  return f.set('roi_area_m2', clipped.area(1));
});

// --------------------------- 11. NODE ATTRIBUTES ------------------------------
var nodes = stack.reduceRegions({
  collection: grid,
  reducer: ee.Reducer.mean(),
  scale: 30,
  crs: UTM_CRS[RUN_CITY],
  tileScale: 8
});

var validAreaImage = ee.Image.pixelArea()
  .updateMask(validMask).rename('valid_area');

nodes = validAreaImage.reduceRegions({
  collection: nodes,
  reducer: ee.Reducer.sum(),
  scale: 30,
  crs: UTM_CRS[RUN_CITY],
  tileScale: 8
}).map(function(f) {
  return f.set('valid_area_m2', f.get('sum'));
});

function addClassArea(fc, classValue, propertyName) {
  var img = ee.Image.pixelArea()
    .updateMask(lulc4.eq(classValue))
    .rename('class_area');
  return img.reduceRegions({
    collection: fc,
    reducer: ee.Reducer.sum(),
    scale: 20,
    crs: UTM_CRS[RUN_CITY],
    tileScale: 8
  }).map(function(f) {
    return f.set(propertyName, f.get('sum'));
  });
}

nodes = addClassArea(nodes, 0, 'water_area');
nodes = addClassArea(nodes, 1, 'built_area');
nodes = addClassArea(nodes, 2, 'vegetation_area');
nodes = addClassArea(nodes, 3, 'bare_area');

nodes = nodes.map(function(f) {
  var roiArea = ee.Number(f.get('roi_area_m2')).max(1);
  var validArea = ee.Number(f.get('valid_area_m2'));

  var water = ee.Number(ee.Algorithms.If(f.get('water_area'), f.get('water_area'), 0));
  var built = ee.Number(ee.Algorithms.If(f.get('built_area'), f.get('built_area'), 0));
  var vegetation = ee.Number(ee.Algorithms.If(f.get('vegetation_area'), f.get('vegetation_area'), 0));
  var bare = ee.Number(ee.Algorithms.If(f.get('bare_area'), f.get('bare_area'), 0));
  var mapped = water.add(built).add(vegetation).add(bare).max(1);

  var centroid = f.geometry().centroid(1)
    .transform('EPSG:4326', 1).coordinates();

  return f.set({
    city: cityName,
    valid_fraction: validArea.divide(roiArea),
    water_pct: water.divide(mapped).multiply(100),
    built_pct: built.divide(mapped).multiply(100),
    vegetation_pct: vegetation.divide(mapped).multiply(100),
    bare_pct: bare.divide(mapped).multiply(100),
    longitude: centroid.get(0),
    latitude: centroid.get(1)
  });
}).filter(ee.Filter.gte('valid_fraction', VALID_COVERAGE_MIN))
  .filter(ee.Filter.notNull([
    'lst','ndvi','ndbi','hri','elevation',
    'water_pct','built_pct','vegetation_pct','bare_pct'
  ]));

print(cityName + ' retained 500 m nodes:', nodes.size());

// --------------------------- 12. LULC AREA SUMMARY ----------------------------
var areaKm2 = ee.Image.pixelArea().divide(1e6);
function classArea(classValue) {
  return areaKm2.updateMask(lulc4.eq(classValue)).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 30,
    maxPixels: 1e13,
    tileScale: 16,
    bestEffort: true
  }).get('area');
}

var areaTable = ee.FeatureCollection([
  ee.Feature(null, {City: cityName, Class: 'Water', Area_km2: classArea(0)}),
  ee.Feature(null, {City: cityName, Class: 'Built-up', Area_km2: classArea(1)}),
  ee.Feature(null, {City: cityName, Class: 'Vegetation', Area_km2: classArea(2)}),
  ee.Feature(null, {City: cityName, Class: 'Bare land', Area_km2: classArea(3)})
]);

// --------------------------- 13. EXPORTS --------------------------------------
Export.table.toDrive({
  collection: nodes,
  description: cityName + '_heat_network_nodes_500m_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: cityName + '_heat_network_nodes_500m_2024',
  fileFormat: 'GeoJSON'
});

Export.table.toDrive({
  collection: areaTable,
  description: cityName + '_LULC_Area_Statistics_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: cityName + '_LULC_Area_Statistics_2024',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: ee.FeatureCollection([
    ee.Feature(null, {
      City: cityName,
      Total_samples: samples.size(),
      Training_samples: trainingSamples.size(),
      Validation_samples: validationSamples.size(),
      Overall_accuracy: cm.accuracy(),
      Kappa: cm.kappa()
    })
  ]),
  description: cityName + '_LULC_Validation_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: cityName + '_LULC_Validation_2024',
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: lulc5,
  description: cityName + '_LULC_2024_RF_5class',
  folder: EXPORT_FOLDER,
  fileNamePrefix: cityName + '_LULC_2024_RF_5class',
  region: roi,
  scale: 30,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: stack.addBands(lulc4.rename('lulc')).toFloat(),
  description: cityName + '_heat_risk_stack_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: cityName + '_heat_risk_stack_2024',
  region: roi,
  scale: 30,
  crs: UTM_CRS[RUN_CITY],
  maxPixels: 1e13
});

// --------------------------- 14. MAP ------------------------------------------
Map.centerObject(cityBoundary, 9);
Map.addLayer(s2, {bands:['B4','B3','B2'], min:0, max:0.3},
             cityName + ' Sentinel-2 2024', false);
Map.addLayer(lulc5, {
  min:0, max:4,
  palette:['1a5bab','d63031','00b14f','98fb98','f1c40f']
}, cityName + ' LULC 5-class', false);
Map.addLayer(hri, {
  min:0, max:1,
  palette:['313695','74add1','ffffbf','f46d43','a50026']
}, cityName + ' HRI', true);

print('FINAL SCRIPT CONFIGURATION: RF trees =', RF_TREES);
print('Run once for each city and start the exports from the Tasks tab.');
