# -*- coding: utf-8 -*-

# --------------------------------------------------
# Pixel size calculations
# --------------------------------------------------

# SEMCL pixel size (scaling from image dimensions)
semcl_pixel_size = (30 / 2056) * (1617 / 341)

# EBSD pixel size (scaling from image dimensions)
ebsd_pixel_size = (5 / 1046) * (4180 / 808)

# --------------------------------------------------
# Area calculations
# --------------------------------------------------

# Number of pixels in each region
semcl_pixel_count = 8553
ebsd_pixel_count = 251 * 198

# Total area estimates
semcl_area = semcl_pixel_size**2 * semcl_pixel_count
ebsd_area = ebsd_pixel_size**2 * ebsd_pixel_count

# --------------------------------------------------
# Output
# --------------------------------------------------

print("EBSD total area:", ebsd_area)
print("EBSD-to-SEMCL pixel size ratio:", ebsd_pixel_size / semcl_pixel_size)
