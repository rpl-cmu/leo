// leocpp variables

class gtsam::Point2;
class gtsam::Pose2;
class gtsam::Vector3;

class gtsam::Point3;
class gtsam::Pose3;
class gtsam::Vector6;

class gtsam::Values;
virtual class gtsam::noiseModel::Base;
virtual class gtsam::NonlinearFactor;
virtual class gtsam::NonlinearFactorGraph;
virtual class gtsam::NoiseModelFactor : gtsam::NonlinearFactor;

namespace leocpp {

#include <leocpp/3rdparty/gpmp2/PlanarSDF.h>
class PlanarSDF {
  PlanarSDF(const gtsam::Point2& origin, double cell_size, const Matrix& data);
  // access
  double getSignedDistance(const gtsam::Point2& point) const;
  void print(string s) const;
};

#include <leocpp/dynamics/QSVelPushMotionRealObjEEFactor.h>
virtual class QSVelPushMotionRealObjEEFactor : gtsam::NoiseModelFactor {
  QSVelPushMotionRealObjEEFactor(size_t objKey0, size_t objKey1, size_t eeKey0, size_t eeKey1,
                                 const double& cSq, const gtsam::noiseModel::Base* noiseModel);
  Vector evaluateError(const gtsam::Pose2& objPose0, const gtsam::Pose2& objPose1,
                       const gtsam::Pose2& eePose0, const gtsam::Pose2& eePose1) const;
};

#include <leocpp/geometry/IntersectionPlanarSDFObjEEFactor.h>
virtual class IntersectionPlanarSDFObjEEFactor : gtsam::NoiseModelFactor {
  IntersectionPlanarSDFObjEEFactor(size_t objKey, size_t eeKey, const PlanarSDF &sdf, const double& eeRadius, const gtsam::noiseModel::Base* noiseModel);
  Vector evaluateError(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose) const;
};

#include <leocpp/contact/TactileRelativeTfPredictionFactor.h>
virtual class TactileRelativeTfPredictionFactor : gtsam::NoiseModelFactor {
  TactileRelativeTfPredictionFactor(size_t objKey1, size_t objKey2, size_t eeKey1, size_t eeKey2,
                                    const Vector& torchModelOutput,
                                    const gtsam::noiseModel::Base* noiseModel);
  Vector evaluateError(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2,
                       const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2) const;
  void setFlags(const bool yawOnlyError,const bool constantModel);
  void setOracle(const bool oracleFactor, const gtsam::Pose2 oraclePose);
  void setLabel(int classLabel, int numClasses);
  // gtsam::Pose2 getMeasTransform();
  gtsam::Pose2 getExpectTransform(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2,
                                  const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2);
};

}  // namespace leocpp
