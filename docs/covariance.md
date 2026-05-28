# Covariance

Covariance is part of the contract, not optional metadata.

## Rules

- pose covariance is 6x6.
- SE(2) demos still use 6x6-compatible covariance.
- covariance must be finite and symmetric within tolerance.
- zero covariance means near-perfect confidence and is almost always invalid.
- graph backends reject relative constraints with zero or non-positive x, y, or yaw variance.
- unknown covariance must be represented as large covariance.
- frame transforms must transform covariance with the appropriate adjoint.
- graph backends convert covariance to information matrices.
- outlier rejection uses Mahalanobis distance.

## Source Quality

Adapters should map source quality into covariance:

- GNSS fix type
- RTK status
- LiDAR registration score
- visual marker reprojection error
- UWB range residual

## Validation Utilities

C++ helpers live in `mrn_core/include/mrn_core/covariance.hpp`:

- `isFiniteCovariance` — rejects NaN/Inf.
- `hasPositiveDiagonal` — rejects zero or negative diagonal entries.
- `isAllZero` — detects the "unknown covariance encoded as zeros" mistake.
- `isSymmetric` — symmetry within absolute and relative tolerance.
- `hasOverconfidentDiagonal` — rejects implausibly small variances.
- `validateCovariance(cov, config)` — returns a `CovarianceValidationStatus`
  enum with names exposed via `validationStatusName`.
- `largeUnknownCovariance(variance)` — produces the recommended large-covariance
  fallback to use instead of zeros when uncertainty is unknown.

`CovarianceValidationConfig::allow_unknown` lets callers explicitly accept the
all-zero case when an upstream contract guarantees no information is available;
default validation rejects it.
