import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Step 1: Open NetCDF file
ds = xr.open_dataset("20240101_prof.nc", decode_times=True, engine="netcdf4")

# Step 2: Extract variables
float_id = ds.PLATFORM_NUMBER.values.astype(str)[0]
lat = ds.LATITUDE.values
lon = ds.LONGITUDE.values
time = ds.JULD.values  # should already be datetime64

pressure = ds.PRES.values
temperature = ds.TEMP.values
salinity = ds.PSAL.values

# Helper: convert time safely
def convert_time(val):
    if np.issubdtype(type(val), np.number):
        return pd.to_datetime("1950-01-01") + pd.to_timedelta(val, unit="D")
    else:
        return pd.to_datetime(val)

# Step 3: Flatten into DataFrame
records = []
for i in range(pressure.shape[0]):
    for j in range(pressure.shape[1]):
        t_val = temperature[i, j]
        s_val = salinity[i, j]
        p_val = pressure[i, j]
        if pd.notna(t_val) and pd.notna(s_val) and pd.notna(p_val):
            records.append({
                "float_id": float_id,
                "profile_number": int(i),
                "time": convert_time(time[i]),
                "lat": float(lat[i]),
                "lon": float(lon[i]),
                "depth": float(p_val),
                "temperature": float(t_val),
                "salinity": float(s_val)
            })

df = pd.DataFrame(records)

# Step 4: Save to SQLite (no setup needed)
engine = create_engine("sqlite:///argo.db")
df.to_sql("argo_profiles", engine, if_exists="append", index=False)

print(f"✅ Successfully inserted {len(df)} rows into argo_profiles table (SQLite: argo.db).")

print(ds.dims)
print(ds.JULD.shape)        # number of profiles
print(ds.LATITUDE.shape)    # should match JULD
print(ds.LONGITUDE.shape)
print(ds.PRES.shape)