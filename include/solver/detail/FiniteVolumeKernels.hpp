#pragma once

#include "grid/Grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace flood::detail::finite_volume {

using State = std::array<double, 3>;

struct FaceFlux {
    std::size_t left = std::numeric_limits<std::size_t>::max();
    std::size_t right = std::numeric_limits<std::size_t>::max();
    State leftFlux{};
    State rightFlux{};
    double massRate = 0.0;
    std::size_t donor = std::numeric_limits<std::size_t>::max();
    bool boundaryOutflow = false;
};

inline FaceFlux rusanovFace(const Cell& left, const Cell& right,
                            std::size_t leftIndex, std::size_t rightIndex,
                            bool xDirection, double gravity, double dryDepth) {
    const double faceBed = std::max(left.bed, right.bed);
    const double leftDepth = std::max(0.0, left.h + left.bed - faceBed);
    const double rightDepth = std::max(0.0, right.h + right.bed - faceBed);
    const double leftScale = left.h > dryDepth ? leftDepth / left.h : 0.0;
    const double rightScale = right.h > dryDepth ? rightDepth / right.h : 0.0;
    State ul{leftDepth, left.hu * leftScale, left.hv * leftScale};
    State ur{rightDepth, right.hu * rightScale, right.hv * rightScale};

    const double ql = xDirection ? ul[1] : ul[2];
    const double qr = xDirection ? ur[1] : ur[2];
    const double vl = leftDepth > dryDepth ? ql / leftDepth : 0.0;
    const double vr = rightDepth > dryDepth ? qr / rightDepth : 0.0;
    State fl{}, fr{};
    if (xDirection) {
        fl = {ul[1], ul[1] * vl + 0.5 * gravity * leftDepth * leftDepth,
              ul[2] * vl};
        fr = {ur[1], ur[1] * vr + 0.5 * gravity * rightDepth * rightDepth,
              ur[2] * vr};
    } else {
        fl = {ul[2], ul[1] * vl, ul[2] * vl + 0.5 * gravity * leftDepth * leftDepth};
        fr = {ur[2], ur[1] * vr, ur[2] * vr + 0.5 * gravity * rightDepth * rightDepth};
    }
    const double waveSpeed = std::max(std::abs(vl) + std::sqrt(gravity * leftDepth),
                                      std::abs(vr) + std::sqrt(gravity * rightDepth));
    State common{};
    for (std::size_t k = 0; k < 3; ++k)
        common[k] = 0.5 * (fl[k] + fr[k] - waveSpeed * (ur[k] - ul[k]));

    FaceFlux face;
    face.left = leftIndex;
    face.right = rightIndex;
    face.leftFlux = common;
    face.rightFlux = common;
    const double leftCorrection = 0.5 * gravity * (left.h * left.h - leftDepth * leftDepth);
    const double rightCorrection = 0.5 * gravity * (right.h * right.h - rightDepth * rightDepth);
    if (xDirection) {
        face.leftFlux[1] += leftCorrection;
        face.rightFlux[1] += rightCorrection;
    } else {
        face.leftFlux[2] += leftCorrection;
        face.rightFlux[2] += rightCorrection;
    }
    face.massRate = common[0];
    return face;
}

inline Cell ghostForOutflow(const Cell& interior, bool xDirection, bool atLowSide) {
    Cell ghost = interior;
    const double normalMomentum = xDirection ? interior.hu : interior.hv;
    const bool pointsOut = atLowSide ? normalMomentum < 0.0 : normalMomentum > 0.0;
    if (!pointsOut) {
        if (xDirection) ghost.hu = -interior.hu;
        else ghost.hv = -interior.hv;
    }
    return ghost;
}

inline Cell ghostForWall(const Cell& interior, bool xDirection) {
    Cell ghost = interior;
    if (xDirection) ghost.hu = -interior.hu;
    else ghost.hv = -interior.hv;
    return ghost;
}

} // namespace flood::detail::finite_volume