# arcpy_gis_service.py
# ArcPy GIS Service for SpatioThematicVLM
# Provides preprocessing and GIS tools (Moran's I, etc.)

import arcpy
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import shutil
import tempfile

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

        # 处理 shp 或 zip
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
    """全局 Moran's I """
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

        # 输出要素类名称
        output_name = f"LocalMorans_{Path(feature_class).stem}"
        output_fc = os.path.join(output_dir, output_name)

        # 可选参数
        conceptualization = data.get("conceptualization", "INVERSE_DISTANCE")
        standardization = data.get("standardization", "ROW")
        apply_fdr = data.get("apply_fdr", True)
        number_of_permutations = data.get("number_of_permutations", 999)

        print(f"[Local Moran's I] 开始分析 → {feature_class} | 字段: {value_field}")
        print(f"[Local Moran's I] 输出路径: {output_fc}")

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
            "message": f"✅ 分析完成！发现 {high_low_outliers_count} 个高低/低高异常区域（HL + LH）"
        })

    except Exception as e:
        print(f"[Local Moran's I] ERROR: {str(e)}")
        raise HTTPException(500, detail=f"Local Moran's I failed: {str(e)}")

@app.post("/gis/reverse_geocode")
async def reverse_geocode(data: dict):
    """对 Local Moran's I 输出图层进行反向地理编码（HH/LL Top 8 + HL/LH 全部 + 完美适配 REV_ 前缀）"""
    try:
        output_fc = data.get("output_feature_class")
        print(f"[Reverse Geocoding Service] 收到请求 → output_feature_class = {output_fc}")

        if not output_fc:
            raise HTTPException(400, detail="output_feature_class is required")

        # 自动补全 .shp
        output_fc = str(output_fc).strip()
        if not output_fc.lower().endswith('.shp'):
            shp_candidate = f"{output_fc}.shp"
            if os.path.exists(shp_candidate):
                output_fc = shp_candidate

        if not os.path.exists(output_fc):
            raise HTTPException(400, detail=f"文件不存在: {output_fc}")

        print(f"[Reverse Geocoding Service] ✅ 找到文件: {output_fc}")

        arcpy.env.overwriteOutput = True
        workspace = os.path.abspath("temp_gis")
        os.makedirs(workspace, exist_ok=True)
        arcpy.env.workspace = workspace
        arcpy.env.scratchWorkspace = workspace

        # 第一层筛选 + 第二层 Top 8 筛选（保持不变）
        significant_layer = "significant_clusters"
        arcpy.management.MakeFeatureLayer(
            in_features=output_fc,
            out_layer=significant_layer,
            where_clause="COType IN ('HH', 'LL', 'HL', 'LH')"
        )

        total_significant = int(arcpy.management.GetCount(significant_layer)[0])
        print(f"[Reverse Geocoding Service] 第一层筛选后共有 {total_significant} 个显著聚类")

        if total_significant == 0:
            return JSONResponse({"status": "success", "locations": {}, "message": "无显著聚类"})

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

        print(f"[Reverse Geocoding Service] 二次筛选后最终处理 {len(selected_oids)} 个最显著要素")

        if not selected_oids:
            return JSONResponse({"status": "success", "locations": {}, "message": "无显著要素"})

        final_layer = "top_significant"
        oid_list = ",".join(map(str, selected_oids))
        arcpy.management.MakeFeatureLayer(
            in_features=significant_layer,
            out_layer=final_layer,
            where_clause=f"{oid_field} IN ({oid_list})"
        )

        # 执行地理编码
        gdb_path = os.path.join(workspace, "reverse_geocode.gdb")
        if not arcpy.Exists(gdb_path):
            arcpy.management.CreateFileGDB(workspace, "reverse_geocode.gdb")

        centroid_fc = os.path.join(gdb_path, "temp_centroids")
        arcpy.management.FeatureToPoint(in_features=final_layer, out_feature_class=centroid_fc, point_location="CENTROID")

        reverse_fc = os.path.join(gdb_path, "reverse_geocoded")
        print(f"[Reverse Geocoding Service] 开始执行 ReverseGeocode（{len(selected_oids)} 个点）...")
        arcpy.geocoding.ReverseGeocode(
            in_features=centroid_fc,
            in_address_locator="https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer",
            out_feature_class=reverse_fc,
            search_distance="500 Meters",
            location_type="ADDRESS_LOCATION"
        )

        # ==================== 关键修复：智能匹配 REV_ 前缀字段 ====================
        print("[Reverse Geocoding Service] ReverseGeocode 完成，正在读取字段...")
        field_list = [f.name for f in arcpy.ListFields(reverse_fc)]
        print(f"[Reverse Geocoding Service] 输出字段列表: {field_list}")

        # 更智能的字段匹配（优先 REV_ 开头的地址字段）
        address_field = None
        for f in field_list:
            if f in ["REV_Match_addr", "REV_LongLabel", "REV_ShortLabel", "Match_addr", "LongLabel", "ShortLabel"]:
                address_field = f
                print(f"[Reverse Geocoding Service] ✅ 找到地址字段: {address_field}")
                break

        # 如果还是没找到，尝试任何包含 "addr" 或 "label" 的字段
        if not address_field:
            for f in field_list:
                if "addr" in f.lower() or "label" in f.lower():
                    address_field = f
                    print(f"[Reverse Geocoding Service] ✅ 找到备选地址字段: {address_field}")
                    break

        if not address_field:
            print("[Reverse Geocoding Service] ⚠️ 仍未找到地址字段，使用坐标兜底")
            address_field = None

        # 读取结果
        locations = {"HH": [], "LL": [], "HL": [], "LH": []}
        cursor_fields = ["COType"]
        if address_field:
            cursor_fields.append(address_field)

        with arcpy.da.SearchCursor(reverse_fc, cursor_fields) as cursor:
            for row in cursor:
                cotype = row[0]
                address = row[1] if address_field and len(row) > 1 else "未知位置"
                if cotype in locations:
                    locations[cotype].append(address)

        for k in locations:
            locations[k] = list(dict.fromkeys(locations[k]))[:3]

        geocoded_info = {
            "status": "success",
            "locations": locations,
            "message": f"已成功获取 {len(selected_oids)} 个最显著聚类位置的地名"
        }

        print(f"[Reverse Geocoding Service] 完成 → {locations}")
        return JSONResponse(geocoded_info)

    except Exception as e:
        print(f"[Reverse Geocoding Service] 未知错误: {str(e)}")
        raise HTTPException(500, detail=f"Reverse Geocoding failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)