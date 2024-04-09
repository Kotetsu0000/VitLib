try:
    from VitLib_cython import *
    USE_CYTHON = True
except:
    from VitLib.VitLib_python import *
    USE_CYTHON = False