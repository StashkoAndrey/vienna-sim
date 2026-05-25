#ifndef BACTERIUM_HPP
#define BACTERIUM_HPP

#include <string>
#include "simulation_stats_6.hpp"

struct UptakeResult {
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
};

class Bacterium {
public:
    float energy;
    int channels_A;
    int channels_B;
    int channels_C;
    std::string type;
    int id;
    bool dead;

    Bacterium(float energy_, int cell_id, int ch_A, int ch_B, int ch_C, const std::string& btype);

    // Returns the actual nutrient uptake used for energy gain in this timestep.
    // The grid must subtract exactly these values from nutrient fields.
    UptakeResult consume(float local_A, float local_B, float local_C);

    bool ready_to_divide() const;

    // Updates StepStats with division/switch events.
    Bacterium* divide(int new_id, StepStats& stats);

private:
    float channel_cost;
    float maintenance_cost;
    float base_metabolic_cost;
    float div_threshold;

    static float build_prob(float local_nutrient, int total_channels);
    bool try_build_channel(char channel_type, float local_nutrient, int total_channels);
    static void record_switch_transition(const std::string& old_type,
                                         const std::string& new_type,
                                         StepStats& stats);
};

#endif // BACTERIUM_HPP
