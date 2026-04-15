# gis_preprocessor_service.py
# ArcPy GIS Preprocessor Service for SpatioThematicVLM
# Supports TIF (raster) and SHP/ZIP (vector) → preview PNG + metadata

import arcpy
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import shutil
import tempfile

app = FastAPI(title="ArcPy GIS Preprocessor - SpatioThematicVLM")

@app.post("/preprocess")
async def preprocess_gis_file(file: UploadFile = File(...)):
    os.makedirs("temp_gis", exist_ok=True)
    temp_path = f"temp_gis/{file.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        output_png = f"temp_gis/preview_{Path(file.filename).stem}.png"
        metadata = {}

        ext = file.filename.lower()

        # 1. 处理栅格（TIF）
        if ext.endswith(('.tif', '.tiff')):
            raster = arcpy.Raster(temp_path)
            metadata = {
                "type": "raster",
                "crs": raster.spatialReference.exportToString() if hasattr(raster, "spatialReference") else "Unknown",
                "extent": str(raster.extent),
                "cellsize": float(raster.meanCellWidth),
                "band_count": raster.bandCount,
                "format": "GeoTIFF"
            }
            arcpy.management.ExportRasterToPNG(temp_path, output_png, "PNG")

        # 2. 处理矢量（SHP 或 ZIP）
        elif ext.endswith(('.shp', '.zip')):
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
                "format": "Shapefile"
            }

            # === 关键修复：使用 arcpy.mp 将矢量渲染成图片 ===
            # 创建临时地图并导出 PNG
            with tempfile.TemporaryDirectory() as tmp_dir:
                aprx_path = os.path.join(tmp_dir, "temp.aprx")
                arcpy.management.CreateMapDocument(aprx_path)  # 创建临时项目
                aprx = arcpy.mp.ArcGISProject(aprx_path)
                m = aprx.listMaps()[0]
                m.addDataFromPath(shp_path)                    # 添加 shapefile 图层
                
                layout = aprx.listLayouts()[0]
                layout.exportToPNG(output_png, resolution=300)

        else:
            raise HTTPException(400, "Unsupported file type")

        return JSONResponse({
            "status": "success",
            "preview_png": output_png,
            "gis_metadata": metadata,
            "raw_path": temp_path
        })

    except Exception as e:
        raise HTTPException(500, detail=f"Preprocessing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)