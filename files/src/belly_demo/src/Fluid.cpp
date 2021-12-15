#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shape.h"
#include "Spring.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"
#include "Fluid.h"
#include "Cloth.h"
#include "Particle.h"

using namespace std;
using namespace Eigen;

Fluid::Fluid(int n, const Eigen::Vector3d& p0, const Eigen::Vector3d& v0, const std::shared_ptr<Cloth> cloth)
{	
	this->n = n;
	Vector3d p_init = p0;
	double unit = 1.0 / (n /25);
	//double unit = 0.1;
	for (int i = 0; i < n; i++) {
		auto p = make_shared<FluidParticle>();
		particles.push_back(p);
		p->i = i;
		p->r = 0.1;
		//p->m = mass / n;
		p->v = v0;
		//p->x = p_init + Vector3d(unit* (i % 5), unit * (i / 25), unit * ((i / 5) % 5));
		p->x = p_init + Vector3d::Random() * 0.1;
		//p->f = Vector3d(0.0, 0.0, 0.0);
	}
	int index = n;
	
	for (auto particle : cloth->particles) {
		auto p = make_shared<FluidParticle>();
		// Push twice to add density
		env_particles.push_back(p);
		p->x = particle->x;
	}
	
	for (auto pos : cloth->subPos) {
		auto p = make_shared<FluidParticle>();
		// Push twice to add density
		env_particles.push_back(p);
		p->x = pos;
	}
	
	/*
	for (int i = 0; i < 100; i++) {
		auto p = make_shared<FluidParticle>();
		// Push twice to add density
		env_particles.push_back(p);
		p->x = p_init + Vector3d(-0.02 * i + 1, -1.5, 0.0);
	}
	*/

	for (int i = 0; i < n; ++i) {
		std::vector < std::shared_ptr<FluidSpring>> temp;
		for (int j = 0; j < n; ++j) {
			temp.push_back(nullptr);
		}
		springs.push_back(temp);
	}
}

Fluid::~Fluid()
{
}



void Fluid::step(double h, const Eigen::Vector3d& grav, const std::shared_ptr<Cloth> cloth)
{
	double r = 0.2;
	double alpha = 0.3;
	double sigma = 0;
	double beta = 0.8;
	double k_spring = 1;

	float k = 1e-2;			   // Far pressure weight
	float k_near = 1e-1;							// Near pressure weight
	float rest_density = 20.0;

	float k_env = 1e-3;			   // Far pressure weight
	float k_near_env = 5e-1;
	float env_rest_density = 1.0;
	double env_r = 1.0;

	
	for (size_t i = 0; i <cloth->particles.size(); i++) {
		env_particles[i]->x = cloth->particles[i]->x;
	}

	size_t n_cloth =cloth->particles.size();
	for (size_t i = 0; i < cloth->subPos.size(); i++) {
		env_particles[n_cloth + i]->x = cloth->subPos[i];
	}
	

	
	for (int i = 0; i < particles.size(); ++i)
	{
		// For each of that particles neighbors
		for (int j : particles[i]->neighbors)
		{
			Vector3d rij = particles[j]->x - particles[i]->x;
			// Get the projection of the velocities onto the vector between them.
			float u = (particles[i]->v - particles[j]->v).transpose() * rij.normalized();
			if (u > 0)
			{
				// Calculate the viscosity impulse between the two particles
				// based on the quadratic function of projected length.
				double l = rij.norm();
				double q = l / r;
				Vector3d I = h * (1 - q) * (sigma * u + beta * u * u) * rij.normalized();

				// Apply the impulses on the two particles
				particles[i]->v -= I * 0.5;
				particles[j]->v += I * 0.5;
			}
		}
	}
	
	for (auto particle : particles) {
		particle->neighbors.clear();
		particle->env_neighbors.clear();
		particle->v += grav * h;
		particle->x0 = particle->x;
		particle->x += particle->v * h;
	}


#pragma omp parallel for
	for (int i = 0; i < particles.size(); ++i)
	{
		// For each of that particles neighbors
		for (int j = 0; j < particles.size(); ++j)
		{
			Vector3d rij = particles[j]->x - particles[i]->x;
			double l = rij.norm();
			double q = l / r;
			if (q < 1 && i != j) {
				particles[i]->neighbors.push_back(j);
			}
		}
	}

	
#pragma omp parallel for
	for (int i = 0; i < particles.size(); ++i)
	{
		// For each of that particles neighbors
		for (int j = 0; j < env_particles.size(); ++j)
		{
			Vector3d rij = env_particles[j]->x - particles[i]->x;
			double l = rij.norm();
			double q = l / env_r;
			if (q < 1) {
				particles[i]->env_neighbors.push_back(j);
			}
		}
	}
	
	
	
	for (int i = 0; i < particles.size(); ++i)
	{
		// For each of that particles neighbors
		for (int j : particles[i]->neighbors)
		{
			Vector3d rij = particles[j]->x - particles[i]->x;
			double l = rij.norm();
			double q = l / r;
			if (springs[i][j] == nullptr) {
				auto s = make_shared<FluidSpring>();
				springs[i][j] = s;
				springs[j][i] = s;
				s->p0 = particles[i];
				s->p0 = particles[j];
				s->L = r;
			}
			double d = 0.2 * springs[i][j]->L;
			if (l > springs[i][j]->L + d) {
				springs[i][j]->L += h * alpha * (l - springs[i][j]->L - d);
			}
			else if (l < springs[i][j]->L - d) {
				springs[i][j]->L -= h * alpha * (springs[i][j]->L - d - l);
			}
		}
	}


	for (int i = 0; i < particles.size(); ++i) {
		for (int j = 0; j < particles.size(); ++j) {
			if (springs[i][j] == nullptr) {
				continue;
			}
			if (springs[i][j]->L > r) {
				springs[i][j] = nullptr;
				springs[j][i] = nullptr;
			}
			Vector3d rij = particles[j]->x - particles[i]->x;
			double l = rij.norm();
			Vector3d D = h * h * k_spring * (1 - springs[j][i]->L / r) * (springs[j][i]->L - l) * rij.normalized();
			particles[i]->x -= D / 2;
			particles[j]->x += D / 2;
		}
	}
	

	for (auto particle : particles) {
		double p = 0;
		double p_near = 0;
		double press = 0;
		double press_near = 0;
		for (int i : particle->neighbors) {
			Vector3d rij = particle->x - particles[i]->x;
			double l = rij.norm();
			double q = l / r;
			if (q < 1) {
				p += (1 - q) * (1 - q);
				p_near += (1 - q) * (1 - q) * (1 - q);
			}
		}

		press = k * (p - rest_density);
		press_near = k_near * p_near;
		//cout << p <<" , "<<press_near << endl;
		Vector3d dx = Vector3d(0.0, 0.0, 0.0);
		for (int i : particle->neighbors) {
			Vector3d rij = particles[i]->x - particle->x;
			double l = rij.norm();
			double q = l / r;
			if (q < 1) {
				Vector3d D = h * h * (press * (1 - q) + press_near * (1 - q) * (1 - q)) * rij.normalized();
				//cout << press * (1-q) <<" , "<< press_near *(1-q) * (1- q) << endl;
				if (i < n) {
					particles[i]->x += D / 2;
				}
				dx -= D / 2;
			}
		}

		for (int i : particle->env_neighbors) {
			Vector3d rij = particle->x - env_particles[i]->x;
			double l = rij.norm();
			double q = l / env_r;
			if (q < 1) {
				p += (1 - q) * (1 - q);
				p_near += (1 - q) * (1 - q) * (1 - q);
			}
		}

		press = k_env * (p - env_rest_density);
		press_near = k_near_env * p_near;
		//cout << p << endl;
		for (int i : particle->env_neighbors) {
			Vector3d rij = env_particles[i]->x - particle->x;
			double l = rij.norm();
			double q = l / env_r;
			if (q < 1) {
				Vector3d D = h * h * (press * (1 - q) + press_near * (1 - q) * (1 - q)) * rij.normalized();
				dx -= D;
			}
		}

		particle->x += dx;
	}

	
	for (auto particle : particles) {
		particle->v = (particle->x - particle->x0) / h;
		
		if (particle->x.y() <= 0) {
			particle->v = Vector3d(particle->v.x(), 0.8 * abs(particle->v.y()), particle->v.z());
			particle->x = Vector3d(particle->x.x(), 0.0, particle->x.z());
		}

		if (particle->x.z() >= 4.0) {
			particle->v = Vector3d(particle->v.x(), particle->v.y(), -abs(particle->v.z()));
		}
		else if (particle->x.z() <= -4.0) {
			particle->v = Vector3d(particle->v.x(), particle->v.y(), abs(particle->v.z()));
		}

		if (particle->x.x() >= 4.0) {
			particle->v = Vector3d(-abs(particle->v.x()), particle->v.y(),particle->v.z());
		}
		else if (particle->x.x() <= -4.0) {
			particle->v = Vector3d(abs(particle->v.x()), particle->v.y(), particle->v.z());
		}
		
		
	}
		

}

void Fluid::draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p, const std::shared_ptr<Shape> sphere) const
{
	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(0.317647, 0.7647, 1.0).data());
	for (auto particle :particles) {
		MV->pushMatrix();
		MV->translate(particle->x.x(), particle->x.y(), particle->x.z());
		MV->scale(0.05);
		glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		sphere->draw(p);
		MV->popMatrix();
	}

	/*
	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(0.117647, 0.5647, 1.0).data());
	for (size_t i = 0; i < env_particles.size(); i++) {
		MV->pushMatrix();
		MV->translate(env_particles[i]->x.x(), env_particles[i]->x.y(), env_particles[i]->x.z());
		MV->scale(0.05);
		glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		sphere->draw(p);
		MV->popMatrix();
	}
	*/
}
