/**
 * @file IntersectionPlanarSDFObjEEFactor.h
 * @brief Intersection factor using 2D signed distance field (vars: obj, ee poses)
 * @author Paloma Sodhi
 */

#ifndef INTERSECTION_PLANAR_SDF_OBJ_EE_FACTOR_H_
#define INTERSECTION_PLANAR_SDF_OBJ_EE_FACTOR_H_

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace logo {

// template <class T>
class IntersectionPlanarSDFObjEEFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2> {
 private:
  PlanarSDF sdf_;
  double eeRadius_;

  bool useAnalyticJacobians_;

 public:
  IntersectionPlanarSDFObjEEFactor(gtsam::Key objKey, gtsam::Key eeKey, const PlanarSDF& sdf, const double& eeRadius, const gtsam::SharedNoiseModel& model = nullptr)
      : NoiseModelFactor2(model, objKey, eeKey), sdf_(sdf), eeRadius_(eeRadius), useAnalyticJacobians_(false) {}

  gtsam::Vector1 IntersectionErrorOneSidedHinge(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose) const {
    double dist, err;
    gtsam::Vector1 errVec;
    gtsam::Point2 eeCenter__obj = objPose.transformTo(gtsam::Point2(eePose.x(), eePose.y()));

    try {
      dist = sdf_.getSignedDistance(eeCenter__obj);
    } catch (SDFQueryOutOfRange&) {
      std::cout << "[IntersectionPlanarSDFFactor] WARNING: SDF query pos (" << eeCenter__obj.x()
                << ", " << eeCenter__obj.y() << ") out of range. Setting error to 0. " << std::endl;
      errVec << 0.0;
      return errVec;
    }

    if (dist > eeRadius_) {
      err = 0;
    } else {
      err = -dist + eeRadius_;
    }
    errVec << err;

    return errVec;
  }

  gtsam::Vector1 IntersectionErrorTwoSidedHinge(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose) const {
    double dist, err;
    gtsam::Vector1 errVec;
    gtsam::Point2 eeCenter__obj = objPose.transformTo(gtsam::Point2(eePose.x(), eePose.y()));

    try {
      dist = sdf_.getSignedDistance(eeCenter__obj);
    } catch (SDFQueryOutOfRange&) {
      std::cout << "[IntersectionPlanarSDFFactor] WARNING: SDF query pos (" << eeCenter__obj.x()
                << ", " << eeCenter__obj.y() << ") out of range. Setting error to 0. " << std::endl;
      errVec << 0.0;
      return errVec;
    }

    if (dist > eeRadius_) {
      err = dist - eeRadius_;
    } else {
      err = -dist + eeRadius_;
    }
    errVec << err;

    return errVec;
  }

  gtsam::Vector1 IntersectionErrorOneSidedHuber(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose) const {
    double dist, err;
    gtsam::Vector1 errVec;
    gtsam::Point2 eeCenter__obj = objPose.transformTo(gtsam::Point2(eePose.x(), eePose.y()));

    try {
      dist = sdf_.getSignedDistance(eeCenter__obj);
    } catch (SDFQueryOutOfRange&) {
      std::cout << "[IntersectionPlanarSDFFactor] WARNING: SDF query pos (" << eeCenter__obj.x()
                << ", " << eeCenter__obj.y() << ") out of range. Setting error to 0. " << std::endl;
      errVec << 0.0;
      return errVec;
    }

    // ref: Ratliff '09
    if (dist >= eeRadius_) {
      err = 0;
    } else if ((dist < eeRadius_) && (dist >= 0)) {  // quadratic
      err = 0.5 * (1 / eeRadius_) * (dist - eeRadius_) * (dist - eeRadius_);
    } else if (dist < 0) {  // linear
      err = -dist + 0.5 * eeRadius_;
    }
    errVec << err;

    return errVec;
  }

  gtsam::Vector1 IntersectionErrorTwoSidedHuber(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose) const {
    double dist, err;
    gtsam::Vector1 errVec;
    gtsam::Point2 eeCenter__obj = objPose.transformTo(gtsam::Point2(eePose.x(), eePose.y()));

    try {
      dist = sdf_.getSignedDistance(eeCenter__obj);
    } catch (SDFQueryOutOfRange&) {
      std::cout << "[IntersectionPlanarSDFFactor] WARNING: SDF query pos (" << eeCenter__obj.x()
                << ", " << eeCenter__obj.y() << ") out of range. Setting error to 0. " << std::endl;
      errVec << 0.0;
      return errVec;
    }

    if (dist >= 2 * eeRadius_) {  // linear
      err = dist - 0.5 * eeRadius_;
    } else if ((dist < 2 * eeRadius_) && (dist >= 0)) {  // quadratic
      err = 0.5 * (1 / eeRadius_) * (dist - eeRadius_) * (dist - eeRadius_);
    } else if (dist < 0) {  // linear
      err = -dist + 0.5 * eeRadius_;
    }
    errVec << err;

    return errVec;
  }

  gtsam::Vector evaluateError(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose,
                       const boost::optional<gtsam::Matrix&> H1 = boost::none,
                       const boost::optional<gtsam::Matrix&> H2 = boost::none) const {
    gtsam::Vector errVec = IntersectionErrorTwoSidedHuber(objPose, eePose);

    gtsam::Matrix J1, J2;
    if (useAnalyticJacobians_) {
      // todo: add analytic derivative
    } else {
      J1 = gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose2>(boost::bind(&IntersectionPlanarSDFObjEEFactor::IntersectionErrorTwoSidedHuber, this, _1, eePose), objPose);
      J2 = gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose2>(boost::bind(&IntersectionPlanarSDFObjEEFactor::IntersectionErrorTwoSidedHuber, this, objPose, _1), eePose);
    }

    if (H1) *H1 = J1;
    if (H2) *H2 = J2;

    return errVec;
  }
};

}  // namespace gtsam

#endif