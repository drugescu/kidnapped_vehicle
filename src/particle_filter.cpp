/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * 
 * Changed by Dragos Rugescu on Dec 12, 2019
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
#include "range.hpp"

using std::string;
using std::vector;
using util::lang::range;
using util::lang::indices;

#define 	NUM_PART 	    7
//#define 	NUM_PART 	    50

#define 	X_INDEX			0
#define 	Y_INDEX			1
#define 	THETA_INDEX		2

#define 	in				:

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
   std::default_random_engine gen;
   
   if (num_particles != NUM_PART) {
	 num_particles = NUM_PART;  // TODO: Set the number of particles
		
	 // GPS measurement uncertainty sigma_pos[3] = [x[m], y[m], theta [rad]] -> std[]
	 auto normal_x  = std::normal_distribution<double>(x, std[X_INDEX]);
	 auto normal_y  = std::normal_distribution<double>(y, std[Y_INDEX]);
	 auto normal_th = std::normal_distribution<double>(theta, std[THETA_INDEX]);
	 
	 // Initialize particles
	 for (auto pi in range(0, num_particles)) {
		 Particle p;
		 p.id = pi;
		 p.x = normal_x(gen);  p.y = normal_y(gen);  p.theta = normal_th(gen);
		 p.weight = 1;
		 
		 // cout << "Particle " << pi << " generated.\n";
		 
		 particles.push_back(p);
	 }

	 is_initialized = true;
   }
  

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
	
	auto normal_x  = std::normal_distribution<double>(0, std_pos[X_INDEX]);
	auto normal_y  = std::normal_distribution<double>(0, std_pos[Y_INDEX]);
	auto normal_th = std::normal_distribution<double>(0, std_pos[THETA_INDEX]);
	
	std::default_random_engine gen;
	
	for (auto& p in particles) {
		if (yaw_rate * yaw_rate <= EPS * EPS) {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else {
			// Add measurements - equations from lesson 5.9 used
			
			// xf = x0 + v/.th * ( sin(th0 + .th*dt) - sin(th0) )
			// yf = y0 + ( sin(th) * x_obs + cos(th) * y_obs )
			// .th = th0 + .th*dt
			//
			// Where .th = theta_dot, derivative of angle with respect to time,
			//		AKA rate of change of angle with respecti to time
			//		in our situation this is the << yaw_rate >>
			//
			
			p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
		}
		
		// Add gaussian noise
		p.x += normal_x(gen);
		p.y += normal_y(gen);
		p.theta += normal_th(gen);
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
   
   for (auto& o : observations) {
		
		auto temp_id = -numeric_limits<int>::max();
		
		// Sort ascending according to distance
		// Not efficient, should keep "o" copies of the predictions 
		//   and keep them sorted, i.e. trace space for time
		auto&& minimum_p = *(std::min_element(
			predicted.begin(),
			predicted.end(),
			[&o]
			(const LandmarkObs& p1, const LandmarkObs& p2)
			{
			     auto dist1 = dist(o.x, o.y, p1.x, p1.y); 
			     auto dist2 = dist(o.x, o.y, p2.x, p2.y);
			     if (dist1 < dist2) return true;
			     return false;
			} ));
		
		// Store information in id field
		if (predicted.size() != 0)
			o.id = minimum_p.id;
		else
			o.id = temp_id;
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
	
	// Select near
	for (auto& p in particles) {
		
		// Find landmarks close to particle
		auto near = *(new vector<LandmarkObs>());
		for (const auto& l in map_landmarks.landmark_list) {
			if(dist(p.x, p.y, l.x_f, l.y_f) <= abs(sensor_range))
				near.push_back( LandmarkObs { l.id_i, l.x_f, l.y_f } );
		}

		// Convert to vehicle space
		auto tf = *(new vector<LandmarkObs>());
		for (auto&& o in observations) {
			tf.push_back( LandmarkObs 
				{ 	o.id, 
					// Homogenous Rot + Trans (eqn 3.33), also lesson 5.17
					(double) o.x * cos(p.theta) - o.y * sin(p.theta) + p.x,
					(double) o.x * sin(p.theta) + o.y * cos(p.theta) + p.y  } );
		}
		
		// dataAssociation - find nearst predictions to each obs
		dataAssociation(near, tf);
		
		// Set weights
		p.weight = 1.0f;
		for(auto m in tf) {
			// Find current obs m.id in the near set - we have stored it in the id field
			auto it = std::find_if(
				near.begin(), 
				near.end(),
				[m](const LandmarkObs& n) { return n.id == m.id; } );
			
			// Get weight and multiply with it
			auto w = (double) weight(m.x, m.y, 
									(*it).x, (*it).y, 
									std_landmark[X_INDEX], 
									std_landmark[Y_INDEX]);
			if (w > EPS) p.weight *= w; else p.weight *= EPS;
			
		}
	}
	
	
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::	 helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   
    std::default_random_engine gen;

  	auto maximum_weight = (double) numeric_limits<double>::min();

    vector<double> weights;

  	// Get max weight 
  	for (auto p in particles) weights.push_back(p.weight);
  	maximum_weight = 
		(*std::max_element(
			particles.begin(), 
			particles.end(),
			[](const Particle& p1, const Particle& p2) { return p1.weight < p2.weight;	} 
		)).weight; 

	auto uniform_w = std::uniform_real_distribution<double> (0.0, maximum_weight);
	auto uniform_i = std::uniform_int_distribution<  int  > (  0, particles.size() - 1);
	
	// Piechart sampling - from robot lesson
	auto beta = 0.0;
	auto index = uniform_i(gen);
	auto resampled = *(new vector<Particle>());

	// Spin wheel, clamp index, add to resampled
	for (auto p in particles) {
		beta += uniform_w(gen);
		while (beta > weights[index]) {
			beta -= weights[index];
			
			// Wrap around
			index = (index + 1) % particles.size();
		}
		
		resampled.push_back(particles[index]);
	}
	
	// Set particles to new arrangement
	particles = resampled;

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
