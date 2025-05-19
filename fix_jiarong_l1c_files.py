import netCDF4 as nc
import os

# Folder paths
input_folder = r'04'
output_folder = os.path.join(input_folder, 'modified')
os.makedirs(output_folder, exist_ok=True)

# File names
source_file = 'awe_l1c_q20_2024106T0214_02248_v01.nc'
target_file = 'awe_l1c_q20_2024315T2303_05500_v01.nc'
output_file = os.path.join(output_folder, source_file)

# Full paths
source_path = os.path.join(input_folder, source_file)
target_path = os.path.join(input_folder, target_file)

# Variables to replace from source
vars_to_replace = ['time', 'Radiance', 'Latitude', 'Longitude', 'Processed_MLCloud']

with nc.Dataset(source_path, 'r') as src, nc.Dataset(target_path, 'r') as tgt:
    time_len = len(src.dimensions['time'])

    with nc.Dataset(output_file, 'w', format='NETCDF4') as new_ds:
        # Copy dimensions, using time from source
        for name, dim in tgt.dimensions.items():
            if name == 'time':
                new_ds.createDimension('time', time_len)
            else:
                new_ds.createDimension(name, len(dim) if not dim.isunlimited() else None)

        # Copy global attributes from target
        new_ds.setncatts({attr: tgt.getncattr(attr) for attr in tgt.ncattrs()})

        # Copy variables
        for name in tgt.variables:
            if name in vars_to_replace:
                # Replace from source
                var_src = src.variables[name]
                chunks = var_src.chunking() if var_src.chunking() else None
                # Separate _FillValue from other attributes
                var_attrs = {attr: var_src.getncattr(attr) for attr in var_src.ncattrs()}
                fill_value = var_attrs.pop('_FillValue', None)
                
                # Create variable (include fill_value only if it exists)
                new_var = new_ds.createVariable(
                    name,
                    var_src.datatype,
                    var_src.dimensions,
                    chunksizes=chunks,
                    zlib=True,
                    complevel=4,
                    fill_value=fill_value
                )
                
                # Assign data
                new_var[:] = var_src[:]
                
                # Assign remaining attributes
                new_var.setncatts(var_attrs)

                print(f"✅ Replaced: {name}")
            else:
                # Copy from target, slicing along 'time' if needed
                var_tgt = tgt.variables[name]
                chunks = var_tgt.chunking() if var_tgt.chunking() else None
                new_var = new_ds.createVariable(
                    name,
                    var_tgt.datatype,
                    var_tgt.dimensions,
                    chunksizes=chunks,
                    zlib=True,
                    complevel=4
                )
                # Determine slicing
                if 'time' in var_tgt.dimensions:
                    slice_idx = var_tgt.dimensions.index('time')
                    slicing = [slice(0, time_len) if dim == 'time' else slice(None) for dim in var_tgt.dimensions]
                    new_var[:] = var_tgt[tuple(slicing)]
                else:
                    new_var[:] = var_tgt[:]
                new_var.setncatts({attr: var_tgt.getncattr(attr) for attr in var_tgt.ncattrs()})
                print(f"Copied (sliced if needed): {name}")

print(f"\n✅ File saved as: {output_file}")
