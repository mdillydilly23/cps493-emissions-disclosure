"""Verify Python ML dependencies (course Python Setup 2026)."""

import matplotlib
import numpy
import pandas
import scipy
import sklearn

print("scipy: %s" % scipy.__version__)
print("numpy: %s" % numpy.__version__)
print("matplotlib: %s" % matplotlib.__version__)
print("pandas: %s" % pandas.__version__)
print("sklearn: %s" % sklearn.__version__)

try:
    import cv2

    print("opencv: %s" % cv2.__version__)
except ImportError:
    print("opencv: not installed")

try:
    import PIL

    print("pillow: %s" % PIL.__version__)
except ImportError:
    print("pillow: not installed")

try:
    import seaborn

    print("seaborn: %s" % seaborn.__version__)
except ImportError:
    print("seaborn: not installed")
