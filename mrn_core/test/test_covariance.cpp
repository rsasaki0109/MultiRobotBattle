#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "mrn_core/covariance.hpp"

namespace
{
using mrn_core::CovarianceValidationConfig;
using mrn_core::CovarianceValidationStatus;
using mrn_core::PoseCovariance6x6;
using mrn_core::covIndex;
using mrn_core::isAllZero;
using mrn_core::isCovarianceValid;
using mrn_core::isFiniteCovariance;
using mrn_core::isSymmetric;
using mrn_core::hasOverconfidentDiagonal;
using mrn_core::hasPositiveDiagonal;
using mrn_core::largeUnknownCovariance;
using mrn_core::validateCovariance;
using mrn_core::validationStatusName;

PoseCovariance6x6 makeDiagonal(double value)
{
  PoseCovariance6x6 covariance{};
  for (std::size_t i = 0; i < 6; ++i) {
    covariance[covIndex(i, i)] = value;
  }
  return covariance;
}
}  // namespace

TEST(MrnCoreCovariance, FiniteAcceptsValid)
{
  EXPECT_TRUE(isFiniteCovariance(makeDiagonal(1.0)));
}

TEST(MrnCoreCovariance, FiniteRejectsNaN)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(0, 0)] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(isFiniteCovariance(cov));
}

TEST(MrnCoreCovariance, FiniteRejectsInf)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(2, 2)] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(isFiniteCovariance(cov));
}

TEST(MrnCoreCovariance, AllZeroDetected)
{
  PoseCovariance6x6 cov{};
  EXPECT_TRUE(isAllZero(cov));
  EXPECT_FALSE(isAllZero(makeDiagonal(1.0)));
}

TEST(MrnCoreCovariance, PositiveDiagonalRequiresAll)
{
  auto cov = makeDiagonal(1.0);
  EXPECT_TRUE(hasPositiveDiagonal(cov));
  cov[covIndex(3, 3)] = 0.0;
  EXPECT_FALSE(hasPositiveDiagonal(cov));
  cov[covIndex(3, 3)] = -0.5;
  EXPECT_FALSE(hasPositiveDiagonal(cov));
}

TEST(MrnCoreCovariance, SymmetricAcceptsDiagonal)
{
  EXPECT_TRUE(isSymmetric(makeDiagonal(1.0)));
}

TEST(MrnCoreCovariance, SymmetricRejectsAsymmetric)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(0, 1)] = 0.5;
  cov[covIndex(1, 0)] = 0.4;
  EXPECT_FALSE(isSymmetric(cov));
}

TEST(MrnCoreCovariance, OverconfidentDetected)
{
  auto cov = makeDiagonal(1e-12);
  EXPECT_TRUE(hasOverconfidentDiagonal(cov));
  EXPECT_FALSE(hasOverconfidentDiagonal(makeDiagonal(1.0)));
}

TEST(MrnCoreCovariance, ValidateAcceptsDefault)
{
  EXPECT_EQ(validateCovariance(makeDiagonal(1.0)), CovarianceValidationStatus::kValid);
  EXPECT_TRUE(isCovarianceValid(makeDiagonal(1.0)));
}

TEST(MrnCoreCovariance, ValidateRejectsNaN)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(0, 0)] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(validateCovariance(cov), CovarianceValidationStatus::kNotFinite);
}

TEST(MrnCoreCovariance, ValidateRejectsInf)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(5, 5)] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(validateCovariance(cov), CovarianceValidationStatus::kNotFinite);
}

TEST(MrnCoreCovariance, ValidateRejectsAllZeroByDefault)
{
  PoseCovariance6x6 cov{};
  EXPECT_EQ(validateCovariance(cov), CovarianceValidationStatus::kAllZero);
}

TEST(MrnCoreCovariance, ValidateAcceptsAllZeroWhenUnknownAllowed)
{
  PoseCovariance6x6 cov{};
  CovarianceValidationConfig config;
  config.allow_unknown = true;
  EXPECT_EQ(validateCovariance(cov, config), CovarianceValidationStatus::kValid);
}

TEST(MrnCoreCovariance, ValidateRejectsNegativeVariance)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(1, 1)] = -2.0;
  EXPECT_EQ(validateCovariance(cov), CovarianceValidationStatus::kNonPositiveDiagonal);
}

TEST(MrnCoreCovariance, ValidateRejectsAsymmetric)
{
  auto cov = makeDiagonal(1.0);
  cov[covIndex(2, 4)] = 0.3;
  cov[covIndex(4, 2)] = -0.3;
  EXPECT_EQ(validateCovariance(cov), CovarianceValidationStatus::kAsymmetric);
}

TEST(MrnCoreCovariance, ValidateRejectsOverconfident)
{
  auto cov = makeDiagonal(1e-12);
  EXPECT_EQ(validateCovariance(cov), CovarianceValidationStatus::kOverconfident);
}

TEST(MrnCoreCovariance, ValidateAcceptsLargeCovariance)
{
  EXPECT_EQ(
    validateCovariance(largeUnknownCovariance(1e9)),
    CovarianceValidationStatus::kValid);
}

TEST(MrnCoreCovariance, LargeUnknownIsSymmetricAndValid)
{
  auto cov = largeUnknownCovariance();
  EXPECT_TRUE(isSymmetric(cov));
  EXPECT_TRUE(hasPositiveDiagonal(cov));
  EXPECT_FALSE(hasOverconfidentDiagonal(cov));
}

TEST(MrnCoreCovariance, ValidationStatusNamesAreStable)
{
  EXPECT_STREQ(validationStatusName(CovarianceValidationStatus::kValid), "valid");
  EXPECT_STREQ(validationStatusName(CovarianceValidationStatus::kNotFinite), "not_finite");
  EXPECT_STREQ(validationStatusName(CovarianceValidationStatus::kAllZero), "all_zero");
  EXPECT_STREQ(
    validationStatusName(CovarianceValidationStatus::kNonPositiveDiagonal),
    "non_positive_diagonal");
  EXPECT_STREQ(validationStatusName(CovarianceValidationStatus::kAsymmetric), "asymmetric");
  EXPECT_STREQ(validationStatusName(CovarianceValidationStatus::kOverconfident), "overconfident");
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
