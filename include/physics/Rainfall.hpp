#pragma once

#include <vector>

namespace flood {

struct RainPoint {
    double timeSeconds;
    double intensityMmPerHour;
};

class Rainfall {
public:
    explicit Rainfall(std::vector<RainPoint> points = {{0.0, 0.0}});
    double metersPerSecond(double timeSeconds) const;
    double nextChangeAfter(double timeSeconds) const;

private:
    std::vector<RainPoint> points_;
};

} // namespace flood