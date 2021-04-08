import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import gtsam
import logo

point2 = gtsam.Point2(0.5, 0.2)
pose2 = gtsam.Pose2(0.8, 0.3, 0.1)

print("point2: {0}".format(point2))
print("pose2: {0}".format(pose2))