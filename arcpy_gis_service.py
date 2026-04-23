# arcpy_gis_service.py
# ArcPy GIS Service for SpatioThematicVLM
# Provides preprocessing and GIS tools (Moran's I, etc.)

import arcpy
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import shutil

app = FastAPI(title="ArcPy GIS Preprocessor & Tool Service")

@app.post("/preprocess")
async def preprocess_gis_file(file: UploadFile = File(...)):
    os.makedirs("temp_gis", exist_ok=True)
    temp_path = f"temp_gis/{file.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        ext = file.filename.lower()
        metadata = {}

        # Process shp or zip
        if ext.endswith(('.shp', '.zip')):
            if ext.endswith('.zip'):
                extract_dir = temp_path + "_extracted"
                shutil.unpack_archive(temp_path, extract_dir)
                shp_files = list(Path(extract_dir).rglob("*.shp"))
                if not shp_files:
                    raise HTTPException(400, "No .shp file found in ZIP")
                shp_path = str(shp_files[0])
            else:
                shp_path = temp_path

            desc = arcpy.Describe(shp_path)
            feature_count = int(arcpy.management.GetCount(shp_path)[0])

            metadata = {
                "type": "vector",
                "geometry_type": desc.shapeType,
                "crs": desc.spatialReference.exportToString() if desc.spatialReference else "Unknown",
                "fields": [f.name for f in desc.fields],
                "feature_count": feature_count,
                "format": "Shapefile",
                "raw_path": shp_path
            }

        else:
            raise HTTPException(400, "Only .shp or .zip files are supported for metadata extraction")

        return JSONResponse({
            "status": "success",
            "gis_metadata": metadata
        })

    except Exception as e:
        raise HTTPException(500, detail=f"Metadata extraction failed: {str(e)}")

@app.post("/gis/global_morans_i")
async def global_morans_i(data: dict):
    """Global Moran's I"""
    try:
        feature_class = data.get("feature_class")
        value_field = data.get("value_field")
        if not feature_class or not value_field:
            raise HTTPException(400, "feature_class and value_field are required")

        print(f"[Global Moran's I] feature_class={feature_class}, value_field={value_field}")

        arcpy.env.overwriteOutput = True

        result = arcpy.stats.SpatialAutocorrelation(
            feature_class,
            value_field,
            "NO_REPORT",  
            "INVERSE_DISTANCE",
            "EUCLIDEAN_DISTANCE",
            "NONE"
        )

        return {
            "tool": "Global Moran's I",
            "result": result[0],
            "z_score": result[1],
            "p_value": result[2],
            "interpretation": result[3]
        }

    except Exception as e:
        print(f"[Global Moran's I] ERROR: {str(e)}")
        raise HTTPException(500, detail=f"Global Moran's I failed: {str(e)}")

@app.post("/gis/local_morans_i")
async def local_morans_i(data: dict):
    """Anselin Local Moran's I"""
    try:
        feature_class = data.get("feature_class")
        value_field = data.get("value_field")
        if not feature_class or not value_field:
            raise HTTPException(400, "feature_class and value_field are required")

        output_dir = os.path.abspath("temp_gis")   
        os.makedirs(output_dir, exist_ok=True)

        arcpy.env.workspace = output_dir
        arcpy.env.overwriteOutput = True
        arcpy.env.scratchWorkspace = output_dir    

        # Output feature class name
        output_name = f"LocalMorans_{Path(feature_class).stem}"
        output_fc = os.path.join(output_dir, output_name)

        # Optional parameters
        conceptualization = data.get("conceptualization", "INVERSE_DISTANCE")
        standardization = data.get("standardization", "ROW")
        apply_fdr = data.get("apply_fdr", True)
        number_of_permutations = data.get("number_of_permutations", 999)

        print(f"[Local Moran's I] Starting analysis → {feature_class} | Field: {value_field}")
        print(f"[Local Moran's I] Output path: {output_fc}")

        arcpy.stats.ClustersOutliers(
            Input_Feature_Class=feature_class,
            Input_Field=value_field,
            Output_Feature_Class=output_name,
            Conceptualization_of_Spatial_Relationships=conceptualization,
            Distance_Method="EUCLIDEAN_DISTANCE",
            Standardization=standardization
        )

        stats = {"HH": 0, "LL": 0, "HL": 0, "LH": 0, "NS": 0}
        with arcpy.da.SearchCursor(output_name, ["COType"]) as cursor:
            for row in cursor:
                cotype = row[0]
                if cotype in stats:
                    stats[cotype] += 1
                else:
                    stats["NS"] += 1

        high_low_outliers_count = stats["HL"] + stats["LH"]

        return JSONResponse({
            "status": "success",
            "tool": "Local Moran's I (Clusters & Outliers)",
            "output_feature_class": output_fc, 
            "output_dir": output_dir,
            "stats": {
                "HH": stats["HH"],
                "LL": stats["LL"],
                "HL": stats["HL"],
                "LH": stats["LH"],
                "NS": stats["NS"]
            },
            "high_low_outliers_count": high_low_outliers_count,
            "message": f"✅ Analysis complete! Found {high_low_outliers_count} high-low/low-high outlier areas (HL + LH)"
        })

    except Exception as e:
        print(f"[Local Moran's I] ERROR: {str(e)}")
        raise HTTPException(500, detail=f"Local Moran's I failed: {str(e)}")

@app.post("/gis/reverse_geocode")
async def reverse_geocode(data: dict):
    """Reverse geocode Local Moran's I output layer (HH/LL Top 8 + HL/LH all + adapted for REV_ prefix)"""
    try:
        output_fc = data.get("output_feature_class")
        print(f"[Reverse Geocoding Service] Received request → output_feature_class = {output_fc}")

        if not output_fc:
            raise HTTPException(400, detail="output_feature_class is required")

        # Auto-complete .shp extension
        output_fc = str(output_fc).strip()
        if not output_fc.lower().endswith('.shp'):
            shp_candidate = f"{output_fc}.shp"
            if os.path.exists(shp_candidate):
                output_fc = shp_candidate

        if not os.path.exists(output_fc):
            raise HTTPException(400, detail=f"File not found: {output_fc}")

        print(f"[Reverse Geocoding Service] ✅ File found: {output_fc}")

        arcpy.env.overwriteOutput = True
        workspace = os.path.abspath("temp_gis")
        os.makedirs(workspace, exist_ok=True)
        arcpy.env.workspace = workspace
        arcpy.env.scratchWorkspace = workspace

        # First filter + second Top 8 filter (keep unchanged)
        significant_layer = "significant_clusters"
        arcpy.management.MakeFeatureLayer(
            in_features=output_fc,
            out_layer=significant_layer,
            where_clause="COType IN ('HH', 'LL', 'HL', 'LH')"
        )

        total_significant = int(arcpy.management.GetCount(significant_layer)[0])
        print(f"[Reverse Geocoding Service] After first filter: {total_significant} significant clusters")

        if total_significant == 0:
            return JSONResponse({"status": "success", "locations": {}, "message": "No significant clusters"})

        desc = arcpy.Describe(significant_layer)
        oid_field = desc.OIDFieldName

        top_oids = {"HH": [], "LL": [], "HL": [], "LH": []}
        with arcpy.da.SearchCursor(significant_layer, [oid_field, "COType", "LMiZScore"]) as cursor:
            for row in cursor:
                oid = row[0]
                cotype = row[1]
                zscore = float(row[2]) if row[2] is not None else 0.0
                top_oids.setdefault(cotype, []).append((abs(zscore), zscore, oid))

        selected_oids = []
        for cotype, items in top_oids.items():
            items.sort(key=lambda x: x[0], reverse=True)
            if cotype in ["HH", "LL"]:
                selected = items[:8]
            else:
                selected = items[:3]
            for _, _, oid in selected:
                selected_oids.append(oid)

        print(f"[Reverse Geocoding Service] After second filter: processing {len(selected_oids)} most significant features")

        if not selected_oids:
            return JSONResponse({"status": "success", "locations": {}, "message": "No significant features"})

        final_layer = "top_significant"
        oid_list = ",".join(map(str, selected_oids))
        arcpy.management.MakeFeatureLayer(
            in_features=significant_layer,
            out_layer=final_layer,
            where_clause=f"{oid_field} IN ({oid_list})"
        )

        # Execute geocoding
        gdb_path = os.path.join(workspace, "reverse_geocode.gdb")
        if not arcpy.Exists(gdb_path):
            arcpy.management.CreateFileGDB(workspace, "reverse_geocode.gdb")

        centroid_fc = os.path.join(gdb_path, "temp_centroids")
        arcpy.management.FeatureToPoint(in_features=final_layer, out_feature_class=centroid_fc, point_location="CENTROID")

        reverse_fc = os.path.join(gdb_path, "reverse_geocoded")
        print(f"[Reverse Geocoding Service] Starting ReverseGeocode ({len(selected_oids)} points)...")
        arcpy.geocoding.ReverseGeocode(
            in_features=centroid_fc,
            in_address_locator="https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer",
            out_feature_class=reverse_fc,
            search_distance="500 Meters",
            location_type="ADDRESS_LOCATION"
        )

        print("[Reverse Geocoding Service] ReverseGeocode complete, reading fields...")
        field_list = [f.name for f in arcpy.ListFields(reverse_fc)]
        print(f"[Reverse Geocoding Service] Output field list: {field_list}")

        # field matching
        address_field = None
        for f in field_list:
            if f in ["REV_Match_addr", "REV_LongLabel", "REV_ShortLabel", "Match_addr", "LongLabel", "ShortLabel"]:
                address_field = f
                break

        # If still not found, try any field containing "addr" or "label"
        if not address_field:
            for f in field_list:
                if "addr" in f.lower() or "label" in f.lower():
                    address_field = f
                    break

        if not address_field:
            print("[Reverse Geocoding Service] ⚠️ Address field still not found, using coordinates as fallback location for all clusters")
            address_field = None

        # Read results
        locations = {"HH": [], "LL": [], "HL": [], "LH": []}
        cursor_fields = ["COType"]
        if address_field:
            cursor_fields.append(address_field)

        with arcpy.da.SearchCursor(reverse_fc, cursor_fields) as cursor:
            for row in cursor:
                cotype = row[0]
                address = row[1] if address_field and len(row) > 1 else "Unknown location"
                if cotype in locations:
                    locations[cotype].append(address)

        for k in locations:
            locations[k] = list(dict.fromkeys(locations[k]))[:3]

        geocoded_info = {
            "status": "success",
            "locations": locations,
            "message": f"Successfully retrieved location names for {len(selected_oids)} most significant clusters"
        }

        print(f"[Reverse Geocoding Service] Complete → {locations}")
        return JSONResponse(geocoded_info)

    except Exception as e:
        print(f"[Reverse Geocoding Service] Unknown error: {str(e)}")
        raise HTTPException(500, detail=f"Reverse Geocoding failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)