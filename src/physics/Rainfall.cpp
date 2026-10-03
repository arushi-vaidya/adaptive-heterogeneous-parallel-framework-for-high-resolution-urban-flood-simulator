#include "physics/Rainfall.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace flood {

Rainfall::Rainfall(std::vector<RainPoint> points) : points_(std::move(points)) {
    if (points_.empty()) throw std::invalid_argument("Rainfall series cannot be empty");
    std::sort(points_.begin(), points_.end(), [](const auto& a, const auto& b) {
        return a.timeSeconds < b.timeSeconds;
    });
    for (std::size_t i = 0; i < points_.size(); ++i) {
        if (!std::isfinite(points_[i].timeSeconds) ||
            !std::isfinite(points_[i].intensityMmPerHour) ||
            points_[i].timeSeconds < 0.0 || points_[i].intensityMmPerHour < 0.0 ||
            (i > 0 && points_[i - 1].timeSeconds == points_[i].timeSeconds)) {
            throw std::invalid_argument("Rainfall times must be unique and values nonnegative");
        }
    }
}

double Rainfall::metersPerSecond(double timeSeconds) const {
    auto upper = std::upper_bound(points_.begin(), points_.end(), timeSeconds,
        [](double time, const RainPoint& point) { return time < point.timeSeconds; });
    const double mmPerHour = upper == points_.begin()
        ? upper->intensityMmPerHour : std::prev(upper)->intensityMmPerHour;
    return mmPerHour / 1000.0 / 3600.0;
}

double Rainfall::nextChangeAfter(double timeSeconds) const {
    const auto next = std::upper_bound(points_.begin(), points_.end(), timeSeconds,
        [](double time, const RainPoint& point) { return time < point.timeSeconds; });
    return next == points_.end() ? points_.back().timeSeconds : next->timeSeconds;
}

} // namespace flood