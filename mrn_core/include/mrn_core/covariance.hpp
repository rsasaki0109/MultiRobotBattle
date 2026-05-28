#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mrn_core
{
using PoseCovariance6x6 = std::array<double, 36>;

inline constexpr std::size_t kPoseCovDim = 6;

inline std::size_t covIndex(std::size_t row, std::size_t col)
{
  return row * kPoseCovDim + col;
}

inline bool isFiniteCovariance(const PoseCovariance6x6 & covariance)
{
  for (const double value : covariance) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

inline bool hasPositiveDiagonal(const PoseCovariance6x6 & covariance)
{
  for (std::size_t i = 0; i < kPoseCovDim; ++i) {
    if (covariance[covIndex(i, i)] <= 0.0) {
      return false;
    }
  }
  return true;
}

inline bool isAllZero(const PoseCovariance6x6 & covariance)
{
  for (const double value : covariance) {
    if (value != 0.0) {
      return false;
    }
  }
  return true;
}

inline bool isSymmetric(
  const PoseCovariance6x6 & covariance,
  double absolute_tolerance = 1e-9,
  double relative_tolerance = 1e-6)
{
  for (std::size_t i = 0; i < kPoseCovDim; ++i) {
    for (std::size_t j = i + 1; j < kPoseCovDim; ++j) {
      const double a = covariance[covIndex(i, j)];
      const double b = covariance[covIndex(j, i)];
      const double diff = std::abs(a - b);
      const double scale = std::max(std::abs(a), std::abs(b));
      if (diff > absolute_tolerance && diff > relative_tolerance * scale) {
        return false;
      }
    }
  }
  return true;
}

inline bool hasOverconfidentDiagonal(
  const PoseCovariance6x6 & covariance,
  double min_variance = 1e-9)
{
  for (std::size_t i = 0; i < kPoseCovDim; ++i) {
    if (covariance[covIndex(i, i)] < min_variance) {
      return true;
    }
  }
  return false;
}

struct CovarianceValidationConfig
{
  bool allow_unknown = false;
  double min_variance = 1e-9;
  double symmetry_absolute_tolerance = 1e-9;
  double symmetry_relative_tolerance = 1e-6;
};

enum class CovarianceValidationStatus
{
  kValid,
  kNotFinite,
  kAllZero,
  kNonPositiveDiagonal,
  kAsymmetric,
  kOverconfident,
};

inline CovarianceValidationStatus validateCovariance(
  const PoseCovariance6x6 & covariance,
  const CovarianceValidationConfig & config = {})
{
  if (!isFiniteCovariance(covariance)) {
    return CovarianceValidationStatus::kNotFinite;
  }
  if (isAllZero(covariance)) {
    return config.allow_unknown ? CovarianceValidationStatus::kValid
                                : CovarianceValidationStatus::kAllZero;
  }
  if (!hasPositiveDiagonal(covariance)) {
    return CovarianceValidationStatus::kNonPositiveDiagonal;
  }
  if (!isSymmetric(
        covariance,
        config.symmetry_absolute_tolerance,
        config.symmetry_relative_tolerance))
  {
    return CovarianceValidationStatus::kAsymmetric;
  }
  if (hasOverconfidentDiagonal(covariance, config.min_variance)) {
    return CovarianceValidationStatus::kOverconfident;
  }
  return CovarianceValidationStatus::kValid;
}

inline bool isCovarianceValid(
  const PoseCovariance6x6 & covariance,
  const CovarianceValidationConfig & config = {})
{
  return validateCovariance(covariance, config) == CovarianceValidationStatus::kValid;
}

inline const char * validationStatusName(CovarianceValidationStatus status)
{
  switch (status) {
    case CovarianceValidationStatus::kValid: return "valid";
    case CovarianceValidationStatus::kNotFinite: return "not_finite";
    case CovarianceValidationStatus::kAllZero: return "all_zero";
    case CovarianceValidationStatus::kNonPositiveDiagonal: return "non_positive_diagonal";
    case CovarianceValidationStatus::kAsymmetric: return "asymmetric";
    case CovarianceValidationStatus::kOverconfident: return "overconfident";
  }
  return "unknown";
}

inline PoseCovariance6x6 largeUnknownCovariance(double variance = 1e6)
{
  PoseCovariance6x6 covariance{};
  for (std::size_t i = 0; i < kPoseCovDim; ++i) {
    covariance[covIndex(i, i)] = variance;
  }
  return covariance;
}

}  // namespace mrn_core
