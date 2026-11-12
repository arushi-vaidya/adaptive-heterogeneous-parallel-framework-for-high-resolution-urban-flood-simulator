#pragma once

// Structure-of-arrays finite-volume kernels shared by the accelerator
// backends (OpenACC and CUDA). Every function is a pure per-cell or per-face
// operation on flat arrays so it can run on a host, inside an OpenACC
// `parallel loop`, or as the body of a CUDA thread. The numerics mirror the
// serial reference in SerialSolver.cpp (hydrostatic reconstruction, Rusanov
// fluxes, donor-cell positivity limiter, rainfall, infiltration and implicit
// Manning friction).

#include <cmath>

#if defined(__CUDACC__)
#define FLOOD_HD __host__ __device__ inline
#elif defined(_OPENACC)
// Kernels are called from inside OpenACC compute regions.
#define FLOOD_HD _Pragma("acc routine seq") inline
#else
#define FLOOD_HD inline
#endif

namespace flood::detail::flat {

constexpr long kNone = -1;

struct Params {
    long rows;
    long cols;
    double dx;
    double dy;
    double gravity;
    double dryDepth;
    double manningN;
    double infiltration;
    int closed; // 1 = reflective walls, 0 = outflow
};

struct Face {
    double leftFlux[3];
    double rightFlux[3];
    double massRate;
    long left;
    long right;
    long donor;
    int boundaryOutflow;
};

struct CellState {
    double bed, h, hu, hv;
};

FLOOD_HD double fmaxd(double a, double b) { return a > b ? a : b; }
FLOOD_HD double fmind(double a, double b) { return a < b ? a : b; }

FLOOD_HD CellState load(const double* bed, const double* h, const double* hu,
                        const double* hv, long i) {
    CellState c;
    c.bed = bed[i];
    c.h = h[i];
    c.hu = hu[i];
    c.hv = hv[i];
    return c;
}

// Ghost cells for closed walls (reflect normal momentum) and outflow
// boundaries (copy, but never allow inflow through the boundary).
FLOOD_HD CellState ghost(CellState interior, bool xDirection, bool atLowSide, int closed) {
    CellState g = interior;
    double& normal = xDirection ? g.hu : g.hv;
    if (closed) {
        normal = -normal;
    } else {
        const bool pointsOut = atLowSide ? normal < 0.0 : normal > 0.0;
        if (!pointsOut) normal = -normal;
    }
    return g;
}

FLOOD_HD void rusanov(const CellState& left, const CellState& right, bool xDirection,
                      double gravity, double dryDepth, Face& face) {
    const double faceBed = fmaxd(left.bed, right.bed);
    const double leftDepth = fmaxd(0.0, left.h + left.bed - faceBed);
    const double rightDepth = fmaxd(0.0, right.h + right.bed - faceBed);
    const double leftScale = left.h > dryDepth ? leftDepth / left.h : 0.0;
    const double rightScale = right.h > dryDepth ? rightDepth / right.h : 0.0;
    const double ul[3] = {leftDepth, left.hu * leftScale, left.hv * leftScale};
    const double ur[3] = {rightDepth, right.hu * rightScale, right.hv * rightScale};
    const double ql = xDirection ? ul[1] : ul[2];
    const double qr = xDirection ? ur[1] : ur[2];
    const double vl = leftDepth > dryDepth ? ql / leftDepth : 0.0;
    const double vr = rightDepth > dryDepth ? qr / rightDepth : 0.0;
    double fl[3], fr[3];
    if (xDirection) {
        fl[0] = ul[1]; fl[1] = ul[1] * vl + 0.5 * gravity * leftDepth * leftDepth; fl[2] = ul[2] * vl;
        fr[0] = ur[1]; fr[1] = ur[1] * vr + 0.5 * gravity * rightDepth * rightDepth; fr[2] = ur[2] * vr;
    } else {
        fl[0] = ul[2]; fl[1] = ul[1] * vl; fl[2] = ul[2] * vl + 0.5 * gravity * leftDepth * leftDepth;
        fr[0] = ur[2]; fr[1] = ur[1] * vr; fr[2] = ur[2] * vr + 0.5 * gravity * rightDepth * rightDepth;
    }
    const double waveSpeed = fmaxd(std::fabs(vl) + std::sqrt(gravity * leftDepth),
                                   std::fabs(vr) + std::sqrt(gravity * rightDepth));
    for (int k = 0; k < 3; ++k) {
        const double common = 0.5 * (fl[k] + fr[k] - waveSpeed * (ur[k] - ul[k]));
        face.leftFlux[k] = common;
        face.rightFlux[k] = common;
    }
    const double leftCorrection = 0.5 * gravity * (left.h * left.h - leftDepth * leftDepth);
    const double rightCorrection = 0.5 * gravity * (right.h * right.h - rightDepth * rightDepth);
    const int normal = xDirection ? 1 : 2;
    face.leftFlux[normal] += leftCorrection;
    face.rightFlux[normal] += rightCorrection;
    face.massRate = face.leftFlux[0];
}

FLOOD_HD void recordDonor(Face& face, bool lowBoundary, bool highBoundary) {
    face.donor = kNone;
    if (face.massRate > 0.0 && face.left != kNone) face.donor = face.left;
    else if (face.massRate < 0.0 && face.right != kNone) face.donor = face.right;
    face.boundaryOutflow = (lowBoundary && face.massRate < 0.0) ||
                           (highBoundary && face.massRate > 0.0);
}

// x-normal face between (row, faceCol-1) and (row, faceCol); index row*(cols+1)+faceCol.
FLOOD_HD void xFace(const Params& p, const double* bed, const double* h, const double* hu,
                    const double* hv, long row, long faceCol, Face& face) {
    const bool low = faceCol == 0, high = faceCol == p.cols;
    face.left = low ? kNone : row * p.cols + faceCol - 1;
    face.right = high ? kNone : row * p.cols + faceCol;
    const CellState left = low ? ghost(load(bed, h, hu, hv, row * p.cols), true, true, p.closed)
                               : load(bed, h, hu, hv, face.left);
    const CellState right = high ? ghost(load(bed, h, hu, hv, row * p.cols + p.cols - 1), true, false, p.closed)
                                 : load(bed, h, hu, hv, face.right);
    rusanov(left, right, true, p.gravity, p.dryDepth, face);
    recordDonor(face, low, high);
}

// y-normal face between (faceRow-1, col) and (faceRow, col); index faceRow*cols+col.
FLOOD_HD void yFace(const Params& p, const double* bed, const double* h, const double* hu,
                    const double* hv, long faceRow, long col, Face& face) {
    const bool low = faceRow == 0, high = faceRow == p.rows;
    face.left = low ? kNone : (faceRow - 1) * p.cols + col;
    face.right = high ? kNone : faceRow * p.cols + col;
    const CellState left = low ? ghost(load(bed, h, hu, hv, col), false, true, p.closed)
                               : load(bed, h, hu, hv, face.left);
    const CellState right = high ? ghost(load(bed, h, hu, hv, (p.rows - 1) * p.cols + col), false, false, p.closed)
                                 : load(bed, h, hu, hv, face.right);
    rusanov(left, right, false, p.gravity, p.dryDepth, face);
    recordDonor(face, low, high);
}

FLOOD_HD double cflRate(const Params& p, double h, double hu, double hv) {
    if (h <= p.dryDepth) return 0.0;
    const double u = hu / h, v = hv / h;
    const double wave = std::sqrt(p.gravity * h);
    return (std::fabs(u) + wave) / p.dx + (std::fabs(v) + wave) / p.dy;
}

FLOOD_HD double outgoingRate(const Params& p, const Face* xf, const Face* yf, long cell) {
    const long row = cell / p.cols, col = cell % p.cols;
    const Face* faces[4] = {&xf[row * (p.cols + 1) + col], &xf[row * (p.cols + 1) + col + 1],
                            &yf[row * p.cols + col], &yf[(row + 1) * p.cols + col]};
    double rate = 0.0;
    for (int k = 0; k < 4; ++k)
        if (faces[k]->donor == cell) rate += std::fabs(faces[k]->massRate);
    return rate;
}

FLOOD_HD double limiterFor(double h, double area, double outgoing, double dt) {
    const double available = h * area;
    return (outgoing * dt > available && outgoing > 0.0) ? available / (outgoing * dt) : 1.0;
}

FLOOD_HD void addFace(const Face& face, bool cellIsRight, double length, double area,
                      const double* limiter, double delta[3]) {
    const double scale = face.donor == kNone ? 1.0 : limiter[face.donor];
    const double* flux = cellIsRight ? face.rightFlux : face.leftFlux;
    const double sign = cellIsRight ? 1.0 : -1.0;
    for (int k = 0; k < 3; ++k) delta[k] += sign * scale * flux[k] * length / area;
}

FLOOD_HD double faceOutflowVolume(const Face& face, const double* limiter, double length, double dt) {
    if (!face.boundaryOutflow) return 0.0;
    const double scale = face.donor == kNone ? 1.0 : limiter[face.donor];
    return scale * std::fabs(face.massRate) * length * dt;
}

// Applies fluxes, rainfall, infiltration and friction to one cell.
// Returns the infiltrated volume; sets invalid when the state became unphysical.
FLOOD_HD double updateCell(const Params& p, const Face* xf, const Face* yf, const double* limiter,
                           long cell, double dt, double rainRate, double* h, double* hu,
                           double* hv, int& invalid) {
    const long row = cell / p.cols, col = cell % p.cols;
    const double area = p.dx * p.dy;
    double delta[3] = {0.0, 0.0, 0.0};
    addFace(xf[row * (p.cols + 1) + col], true, p.dy, area, limiter, delta);
    addFace(xf[row * (p.cols + 1) + col + 1], false, p.dy, area, limiter, delta);
    addFace(yf[row * p.cols + col], true, p.dx, area, limiter, delta);
    addFace(yf[(row + 1) * p.cols + col], false, p.dx, area, limiter, delta);
    double depth = h[cell] + dt * (delta[0] + rainRate);
    double qx = hu[cell] + dt * delta[1];
    double qy = hv[cell] + dt * delta[2];
    if (depth < 0.0 && depth > -1e-12) depth = 0.0;
    if (depth < 0.0 || !std::isfinite(depth) || !std::isfinite(qx) || !std::isfinite(qy)) {
        invalid = 1;
        h[cell] = depth; hu[cell] = qx; hv[cell] = qy;
        return 0.0;
    }
    invalid = 0;
    const double infiltrated = fmind(depth, p.infiltration * dt);
    depth -= infiltrated;
    if (depth <= p.dryDepth) {
        qx = 0.0;
        qy = 0.0;
    } else if (p.manningN > 0.0) {
        const double u = qx / depth, v = qy / depth;
        const double speed = std::hypot(u, v);
        const double coefficient = p.gravity * p.manningN * p.manningN * speed /
                                   std::pow(depth, 4.0 / 3.0);
        const double frictionScale = 1.0 / (1.0 + dt * coefficient);
        qx *= frictionScale;
        qy *= frictionScale;
    }
    h[cell] = depth;
    hu[cell] = qx;
    hv[cell] = qy;
    return infiltrated * area;
}

} // namespace flood::detail::flat
