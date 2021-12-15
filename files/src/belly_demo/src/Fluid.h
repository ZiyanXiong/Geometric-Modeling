#pragma once
#ifndef Fluid_H
#define Fluid_H

#include <vector>
#include <memory>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Spring;
class MatrixStack;
class Program;
class Shape;
class Cloth;

struct FluidParticle
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	double r; // radius
	double m; // mass
	int i;  // starting index
	Eigen::Vector3d x0; // initial position
	Eigen::Vector3d x;  // position
	Eigen::Vector3d v;  // velocity
	Eigen::Vector3d f;  // force
	std::vector<int> neighbors;
	std::vector<int> env_neighbors;
};

struct FluidSpring
{
	std::shared_ptr<FluidParticle> p0;
	std::shared_ptr<FluidParticle> p1;
	double E;
	double L;
};

class Fluid
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		Fluid(int n,
			const Eigen::Vector3d& p0,
			const Eigen::Vector3d& v0,
			const std::shared_ptr<Cloth> cloth
		);
	virtual ~Fluid();

	void step(double h, const Eigen::Vector3d& grav, const std::shared_ptr<Cloth> cloth);

	//void init();
	void draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p, const std::shared_ptr<Shape> sphere) const;
	std::vector< std::shared_ptr<FluidParticle> > particles;
	std::vector< std::shared_ptr<FluidParticle> > env_particles;
private:
	int n;
	std::vector<std::vector< std::shared_ptr<FluidSpring> >> springs;

};

#endif
