#include <iostream>

#include "Scene.h"
#include "Particle.h"
#include "Cloth.h"
#include "Shape.h"
#include "Program.h"
#include "Fluid.h"

using namespace std;
using namespace Eigen;

Scene::Scene() :
	t(0.0),
	h(1e-2),
	grav(0.0, 0.0, 0.0)
{
}

Scene::~Scene()
{
}

void Scene::load(const string &RESOURCE_DIR)
{
	// Units: meters, kilograms, seconds
	h = 5e-3;
	
	grav << 0.0, -9.8, 0.0;
	
	int rows = 25;
	int cols = 25;
	double mass = 0.1;
	double stiffness = 1;
	Vector3d x00(-1.0, 3.0, 0.0);
	Vector3d x01(1.0, 3.0, 0.0);
	Vector3d x10(-1.0, 1.0, 0.0);
	Vector3d x11(1.0, 1.0, 0.0);

	cloth = make_shared<Cloth>(rows, cols, x00, x01, x10, x11, mass, stiffness);
	fluid = make_shared<Fluid>(200, Vector3d(0.0, 1.5, 0.4), Vector3d (0.0, 0.0, 0.0), cloth);
	sphereShape = make_shared<Shape>();
	sphereShape->loadMesh(RESOURCE_DIR + "sphere2.obj");
	
	auto sphere = make_shared<Particle>(sphereShape);
	spheres.push_back(sphere);
	sphere->r = 0.1;
	sphere->x = Vector3d(0.0, 0.2, 0.0);
}

void Scene::init()
{
	sphereShape->init();
	cloth->init();
}

void Scene::tare()
{
	for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->tare();
	}
	cloth->tare();
}

void Scene::reset()
{
	t = 0.0;
	for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->reset();
	}
	cloth->reset();
}

void Scene::step(bool* keys)
{
	t += h;
	/*
	// Move the sphere
	if(!spheres.empty()) {
		auto s = spheres.front();
		Vector3d x0 = s->x;
		s->x(2) = 0.5 * sin(0.5*t);
	}
	*/
	// Simulate the cloth
	cloth->step(h, grav, fluid, keys);
	fluid->step(h, grav, cloth);
}

void Scene::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> prog) const
{	/*
	glUniform3fv(prog->getUniform("kdFront"), 1, Vector3f(1.0, 1.0, 1.0).data());
	for(int i = 0; i < (int)spheres.size(); ++i) {
		//spheres[i]->draw(MV, prog);
	}
	*/
	cloth->draw(MV, prog);
	fluid->draw(MV, prog, sphereShape);
}
