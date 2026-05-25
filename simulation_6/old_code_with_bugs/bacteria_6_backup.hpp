#ifndef BACTERIUM_HPP
#define BACTERIUM_HPP

#include <string>

class Bacterium {
public:
    float energy;
    int channels_A;
    int channels_B;
    int channels_C;
    std::string type;
    int id;
    bool dead;

    // Constructor
    Bacterium(float energy_, int cell_id, int ch_A, int ch_B, int ch_C, const std::string& btype);

    // Core behavior
    void consume(float local_A, float local_B, float local_C);
    bool ready_to_divide() const;
    Bacterium* divide(int new_id);

private:
    // Cached constants
    float channel_cost;
    float maintenance_cost;
    float base_metabolic_cost;
    float div_threshold;

    static float build_prob(float local_nutrient, int total_channels);
};

#endif // BACTERIUM_HPP
