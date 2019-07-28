/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::numeric_limits;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 15;  // TODO: Set the number of particles

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        this->particles.push_back(particle);
        this->weights.push_back(1.0);
    }

    this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

    for (auto &particle : this->particles) {
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;

        double pred_x, pred_y, pred_theta;

        if (fabs(yaw_rate) > 1e-3) {
            pred_x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
            pred_y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
            pred_theta = theta + yaw_rate * delta_t;
        } else {
            pred_x = x + velocity * cos(theta) * delta_t;
            pred_y = y + velocity * sin(theta) * delta_t;
            pred_theta = theta;
        }

        normal_distribution<double> dist_x(pred_x, std_pos[0]);
        normal_distribution<double> dist_y(pred_y, std_pos[1]);
        normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */

    for (auto &obs : observations) {
        double min_dist = numeric_limits<double>::max();
        int min_id = -1;

        for (auto &pred : predicted) {
            double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                min_id = pred.id;
            }
        }

        obs.id = min_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    double s_x = std_landmark[0];
    double s_y = std_landmark[1];
    double s_x_squared = pow(s_x, 2);
    double s_y_squared = pow(s_y, 2);
    double gaussian_const = 1.0 / (2.0 * M_PI * s_x * s_y);

    vector<double> updated_weights;

    for (auto &particle : this->particles) {
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;

        vector<LandmarkObs> valid_landmarks;
        for (auto &lm : map_landmarks.landmark_list) {
            float lm_x = lm.x_f;
            float lm_y = lm.y_f;
            int lm_id = lm.id_i;

            if (fabs(x - lm_x) <= sensor_range && fabs(y - lm_y) <= sensor_range) {
                valid_landmarks.push_back(LandmarkObs { lm_id, lm_x, lm_y });
            }
        }

        vector<LandmarkObs> transformed_obs;
        for (auto &obs : observations) {
            double trans_x = x + cos(theta) * obs.x - sin(theta) * obs.y;
            double trans_y = y + sin(theta) * obs.x + cos(theta) * obs.y;

            transformed_obs.push_back(LandmarkObs { obs.id, trans_x, trans_y });
        }

        this->dataAssociation(valid_landmarks, transformed_obs);

        particle.weight = 1.0;
        for (auto &trans_ob : transformed_obs) {

            for (auto &v_lm : valid_landmarks) {
                if (trans_ob.id == v_lm.id) {
                    double prob = gaussian_const
                                  * exp(-1.0 * ((pow((trans_ob.x - v_lm.x), 2) / (2.0 * s_x_squared))
                                                + pow((trans_ob.y - v_lm.y), 2)/(2.0 * s_y_squared)));

                    particle.weight *= prob;
                }
            }
        }

        updated_weights.push_back(particle.weight);
    }

    this->weights = updated_weights;
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    vector<Particle> resampled_particles;

    // pick a random index
    uniform_int_distribution<int> rand_discrete(0, num_particles-1);
    int index = rand_discrete(gen);

    // get max weight
    double max_weight = *max_element(this->weights.begin(), this->weights.end());

    // uniform random distribution [0.0, max_weight)
    uniform_real_distribution<double> rand_continuous(0.0, max_weight);

    // Spinning wheel re-sampling
    double beta = 0.0;
    for (int i = 0; i < num_particles; i++) {
        beta += rand_continuous(gen) * 2.0;
        while (beta > this->weights[index]) {
            index = (index + 1) % num_particles;
            beta -= this->weights[index];
        }
        resampled_particles.push_back(particles[index]);
    }

    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}