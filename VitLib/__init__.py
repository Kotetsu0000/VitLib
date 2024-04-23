from VitLib.image_processing import *
try:
    from VitLib.VitLib_cython import *
    from VitLib.VitLib_cython.membrane import *
    from VitLib.VitLib_cython.nucleus import *
    USE_CYTHON = True
except:
    from VitLib.VitLib_python import *
    from VitLib.VitLib_python.membrane import *
    from VitLib.VitLib_python.nucleus import *
    USE_CYTHON = False