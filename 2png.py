import os
from PIL import Image

parentfolder= "C:\\Users\\leonk\\OneDrive\\Projekt_3D_ Druck_KI_Studium\\03_Datensatz"
subfolder = "Original"
newsubfolder = "Konvertiert"
extensions = [".png", ".PNG", ".jpg", ".JPG", ".bmp", ".BMP"]

counter = 0
for path, dirs, files in os.walk(os.path.join(parentfolder, subfolder)):
    for dirname in dirs:
        dirpath = path.replace(os.path.join(parentfolder,subfolder),"")
        dirpath = parentfolder + "\\" + newsubfolder + dirpath + "\\" + dirname
        if os.path.exists(dirpath):
            print(f'Directory {dirpath} already exists, skipping.')
        else:
            os.mkdir(dirpath)
            print(f'Created Directory {dirpath}')

    for file in files:
        for ext in extensions:
            if ext in file:
                filename = os.path.join(path, file)
                newfilename = filename.replace(subfolder, newsubfolder).replace(ext, ".png")
                if os.path.exists(newfilename):
                    print(f'{newfilename.split(parentfolder)[1]} already exists, skipping.')
                else:
                    print(f'Converting {filename.split(parentfolder)[1]} to {newfilename.split(parentfolder)[1]}')
                    Image.open(filename).save(newfilename)
                counter += 1
                print(f'Done ({counter})')
                break


print(f"Done. Converted {counter} files.")

# Supported file formats
# .blp
# .bmp
# .bufr
# .cur
# .dcx
# .dds
# .dib
# .eps, .ps
# .fit, .fits
# .flc, .fli
# .ftc, .ftu
# .gbr
# .gif
# .grib
# .h5, .hdf
# .icns
# .ico
# .im
# .iim
# .jfif, .jpe, .jpeg, .jpg
# .j2c, .j2k, .jp2, .jpc, .jpf, .jpx
# .mpeg, .mpg
# .msp
# .pcd
# .pcx
# .pxr
# .apng, .png
# .pbm, .pgm, .pnm, .ppm
# .psd
# .bw, .rgb, .rgba, .sgi
# .ras
# .icb, .tga, .vda, .vst
# .tif, .tiff
# .webp
# .emf, .wmf
# .xbm
# .xpm
