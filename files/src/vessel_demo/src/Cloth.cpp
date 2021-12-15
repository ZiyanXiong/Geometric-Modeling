#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Cloth.h"
#include "Particle.h"
#include "Spring.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"
#include "Fluid.h"

using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> T;

shared_ptr<Spring> createSpring(const shared_ptr<Particle> p0, const shared_ptr<Particle> p1, double E)
{
	auto s = make_shared<Spring>(p0, p1);
	s->E = E;
	Vector3d x0 = p0->x;
	Vector3d x1 = p1->x;
	Vector3d dx = x1 - x0;
	s->L = dx.norm();
	return s;
}

Cloth::Cloth(int rows, int cols,
	const Vector3d& x00,
	const Vector3d& x01,
	const Vector3d& x10,
	const Vector3d& x11,
	double mass,
	double stiffness)
{
	assert(rows > 1);
	assert(cols > 1);
	assert(mass > 0.0);
	assert(stiffness > 0.0);

	this->rows = rows;
	this->cols = cols;

	// Create particles
	n = 0;
	double r = 0.02; // Used for collisions
	double cylinder_r = x10.norm();
	int nVerts = rows * cols;
	for (int i = 0; i < rows; ++i) {
		double u = i * 1.0 / rows;
		Vector3d offset = cylinder_r * Vector3d(0.0f, sin(u * 2 * glm::pi<float>()), cos(u * 2 * glm::pi<float>()));
		for (int j = 0; j < cols; ++j) {
			double v = j / (cols - 1.0);
			Vector3d x = (1 - v) * x00 + v * x01 + offset;
			auto p = make_shared<Particle>();
			subPos.push_back(x + 0.2 * offset);
			particles.push_back(p);
			p->r = r;
			p->x = x;
			p->v << 0.0, 0.0, 0.0;
			p->m = mass / (nVerts);
			p->fixed = false;
			p->i = n;
			n += 3;

		}
	}

	for (int i = 0; i < rows / 2; i++) {
		int p0 = i * cols;
		int p1 = (i + rows / 2) * cols;
		int p2 = i * cols + cols - 1;
		int p3 = (i + rows / 2) * +cols - 1;
		for (int j = 1; j < 10; j++) {
			subPos.push_back((j / 10.0) * subPos[p0] + (1 - j / 10.0) * subPos[p1]);
			//subPos.push_back((j / 5.0) * subPos[p2] + (1 - j / 5.0) * subPos[p3]);
		}
	}

	// Create x springs
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols - 1; ++j) {
			int k0 = i * cols + j;
			int k1 = k0 + 1;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}

	// Create y springs
	for (int j = 0; j < cols; ++j) {
		for (int i = 0; i < rows - 1; ++i) {
			int k0 = i * cols + j;
			int k1 = k0 + cols;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}

	// Create shear springs
	for (int i = 0; i < rows - 1; ++i) {
		for (int j = 0; j < cols - 1; ++j) {
			int k00 = i * cols + j;
			int k10 = k00 + 1;
			int k01 = k00 + cols;
			int k11 = k01 + 1;
			springs.push_back(createSpring(particles[k00], particles[k11], stiffness));
			springs.push_back(createSpring(particles[k10], particles[k01], stiffness));
		}
	}

	// Create x bending springs
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols - 2; ++j) {
			int k0 = i * cols + j;
			int k2 = k0 + 2;
			springs.push_back(createSpring(particles[k0], particles[k2], stiffness));
		}
	}

	// Create y bending springs
	for (int j = 0; j < cols; ++j) {
		for (int i = 0; i < rows - 2; ++i) {
			int k0 = i * cols + j;
			int k2 = k0 + 2 * cols;
			springs.push_back(createSpring(particles[k0], particles[k2], stiffness));
		}
	}
	
	// Create Seam springs
	for (int j = 0; j < cols; ++j) {
		int k0 = 0 * cols + j;
		int k1 = (rows - 1) * cols + j;
		int k2 = cols + j;
		int k3 = (rows - 2) * cols + j;
		springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		springs.push_back(createSpring(particles[k0], particles[k2], stiffness));
		springs.push_back(createSpring(particles[k1], particles[k3], stiffness));

	}
	
	for (int j = 0; j < cols-1; ++j) {
		int k0 = 0 * cols + j;
		int k1 = (rows - 1) * cols + j + 1;
		int k2 = (rows - 1) * cols + j;
		int k3 = 0 * cols + j + 1;
		springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		springs.push_back(createSpring(particles[k2], particles[k3], stiffness));
	}
	
	// Build system matrices and vectors
	M.resize(n, n);
	K.resize(n, n);
	v.resize(n);
	f.resize(n);

	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();
	posBuf.resize(nVerts * 3);
	norBuf.resize(nVerts * 3);
	updatePosNor();

	// Texture coordinates (don't change)
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			texBuf.push_back(i / (rows - 1.0));
			texBuf.push_back(j / (cols - 1.0));
		}
	}

	// Elements (don't change)
	for (int i = 0; i < rows - 1; ++i) {
		for (int j = 0; j < cols; ++j) {
			int k0 = i * cols + j;
			int k1 = k0 + cols;
			// Triangle strip
			eleBuf.push_back(k0);
			eleBuf.push_back(k1);
		}
	}
}

Cloth::~Cloth()
{
}

void Cloth::tare()
{
	for(int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->tare();
	}
}

void Cloth::reset()
{
	for(int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->reset();
	}
	updatePosNor();
}

void Cloth::updatePosNor()
{
	/*
	subPos.clear();

	for (int i = 0; i < rows / 2; i++) {
		int p0 = i * cols;
		int p1 = (i + rows / 2) * cols;
		int p2 = i * cols + cols - 1;
		int p3 = (i + rows / 2) * + cols - 1;
		for(int j = 1; j < 5; j++){
			//subPos.push_back((j / 5.0) * particles[p0]->x + (1 - j / 5.0) * particles[p1]->x);
			//subPos.push_back((j / 5.0) * particles[p2]->x + (1 - j / 5.0) * particles[p3]->x);
		}
	}
	*/

	// Position
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int k = i * cols + j;
			Vector3d x = particles[k]->x;
			posBuf[3 * k + 0] = x(0);
			posBuf[3 * k + 1] = x(1);
			posBuf[3 * k + 2] = x(2);
		}
	}

	// Normal
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// Each particle has four neighbors
			//
			//      v1
			//     /|\
			// u0 /_|_\ u1
			//    \ | /
			//     \|/
			//      v0
			//
			// Use these four triangles to compute the normal
			int k = i * cols + j;
			int ku0 = k - 1;
			int ku1 = k + 1;
			int kv0 = k - cols;
			int kv1 = k + cols;
			Vector3d x = particles[k]->x;
			Vector3d xu0, xu1, xv0, xv1, dx0, dx1, c;
			Vector3d nor(0.0, 0.0, 0.0);
			int count = 0;
			// Top-right triangle
			if (j != cols - 1 && i != rows - 1) {
				xu1 = particles[ku1]->x;
				xv1 = particles[kv1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Top-left triangle
			if (j != 0 && i != rows - 1) {
				xu1 = particles[kv1]->x;
				xv1 = particles[ku0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-left triangle
			if (j != 0 && i != 0) {
				xu1 = particles[ku0]->x;
				xv1 = particles[kv0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-right triangle
			if (j != cols - 1 && i != 0) {
				xu1 = particles[kv0]->x;
				xv1 = particles[ku1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			nor /= count;
			nor.normalize();
			norBuf[3 * k + 0] = nor(0);
			norBuf[3 * k + 1] = nor(1);
			norBuf[3 * k + 2] = nor(2);
		}
	}
}


void fill_sparse(vector<T>& A, Matrix3d Ks, int i, int j ,bool diagonal) {
	if (diagonal) {
		A.push_back(T(i, j, Ks(0, 0)));
		A.push_back(T(i + 1, j + 1, Ks(1, 1)));
		A.push_back(T(i + 2, j + 2, Ks(2, 2)));
	}
	else {
		for (int k = 0; k < 9; k++) {
			A.push_back(T(i + k / 3, j + k % 3, Ks(k / 3, k % 3)));
		}
	}
}


void Cloth::step(double h, const Vector3d &grav, const shared_ptr<Fluid> fluid, bool* keys)
{
	M.setZero();
	K.setZero();
	v.setZero();
	f.setZero();

	SparseMatrix<double> K_sparse(n, n);
	SparseMatrix<double> M_sparse(n, n);
	vector<T> A;
	vector<T> B;
	//
	// IMPLEMENT ME!
	//

	for (auto particle: particles) {
		if (particle->i >= 0) {
			M.diagonal().segment<3>(particle->i) << particle->m, particle->m, particle->m;
			fill_sparse(B, MatrixXd::Identity(3, 3) * particle->m, particle->i, particle->i, true);
			v.segment<3>(particle->i) = particle->v;
			//f.segment<3>(particle->i) = grav * particle->m;
			f.segment<3>(particle->i) = grav * 0.0f;
		}
	}

	for (auto particle : particles) {

		if (particle->x.z() >= 4.0) {
			v.segment<3>(particle->i) = Vector3d(v.segment<3>(particle->i).x(), v.segment<3>(particle->i).y(), -0.5 * abs(v.segment<3>(particle->i).z()));
		}
		else if (particle->x.z() <= -4.0) {
			v.segment<3>(particle->i) = Vector3d(v.segment<3>(particle->i).x(), v.segment<3>(particle->i).y(), 0.5 *abs(v.segment<3>(particle->i).z()));
		}

		if (particle->x.x() >= 4.0) {
			v.segment<3>(particle->i) = Vector3d(- 0.5 * abs(v.segment<3>(particle->i).x()), v.segment<3>(particle->i).y(), v.segment<3>(particle->i).z());
		}
		else if (particle->x.x() <= -4.0) {
			v.segment<3>(particle->i) = Vector3d( 0.5 * abs(v.segment<3>(particle->i).x()), v.segment<3>(particle->i).y(), v.segment<3>(particle->i).z());
		}
	}
	
	for (auto spring : springs) {

		Vector3d fs = spring->E * ((spring->p0->x - spring->p1->x).norm() - spring->L) * (spring->p1->x - spring->p0->x).normalized();
		if (spring->p0->i >= 0) {
			f.segment<3>(spring->p0->i) += fs;
		}
		if (spring->p1->i >= 0) {
			f.segment<3>(spring->p1->i) -= fs;
		}
	}
	
	Matrix3d Ks;
	for (auto spring : springs) {
		int i = spring->p0->i;
		int j = spring->p1->i;
		double l = (spring->p0->x - spring->p1->x).norm();
		Vector3d delta_x = spring->p1->x - spring->p0->x;
		Ks = spring->E / (l * l) * ((1 - (l - spring->L) / spring->L) * delta_x * delta_x.transpose() + (l - spring->L) / spring->L * delta_x.transpose() * delta_x * MatrixXd::Identity(3, 3));
		if (i >= 0) {
			K.block<3, 3>(i, i) += -Ks;
			fill_sparse(A, -Ks, i, i, false);
		}
		if (i >= 0 && j >= 0) {
			K.block<3, 3>(i, j) += Ks;
			K.block<3, 3>(j, i) += Ks;
			fill_sparse(A, Ks, i, j, false);
			fill_sparse(A, Ks, j, i, false);
		}
		if (j >= 0) {
			K.block<3, 3>(j, j) += -Ks;
			fill_sparse(A, -Ks, j, j, false);
		}
	}

	double c = 5e-4;
	
	for (auto particle : particles) {
		if (particle->i > 0 && particle->x.y() >0) {
			for (auto sphere : fluid->particles) {
				Vector3d delta_x = particle->x - sphere->x;
				double d = 0.75 +  sphere->r - delta_x.norm();
				if (d > 0 && particle->i != sphere->i ) {
					f.segment<3>(particle->i) += c * d * delta_x.normalized();
					K.block<3, 3>(particle->i, particle->i) += c * d * MatrixXd::Identity(3, 3);
					fill_sparse(A, c * d * MatrixXd::Identity(3, 3), particle->i, particle->i, true);
				}
			}
		}
	}
	
	/*
	size_t end_i = fluid->env_particles.size() - 1;
	c = 5;
	for (auto particle : particles) {
		if (particle->i > 0) {
			for (size_t i = 0; i < 100; i++) {
				Vector3d delta_x = particle->x - fluid->env_particles[end_i - i]->x;
				double d = 0.3 + particle->r + fluid->env_particles[end_i - i]->r - delta_x.norm();
				if (d > 0) {
					f.segment<3>(particle->i) += c * d * delta_x.normalized();
					K.block<3, 3>(particle->i, particle->i) += c * d * MatrixXd::Identity(3, 3);
					fill_sparse(A, c * d * MatrixXd::Identity(3, 3), particle->i, particle->i, true);
				}
			}
		}
	}
	*/

	VectorXd x;
	if (keys['1']) {
		x = M.ldlt().solve(M * v + h * f);
	}
	else if (keys['2']) {
		x = (M - h * h * K).ldlt().solve(M * v + h * f);
	}
	else {
		K_sparse.setFromTriplets(A.begin(), A.end());
		M_sparse.setFromTriplets(B.begin(), B.end());
		ConjugateGradient< SparseMatrix<double> > cg;
		cg.setMaxIterations(25);
		cg.setTolerance(1e-6);
		cg.compute(M_sparse - h * h * K_sparse);
		x = cg.solveWithGuess(M_sparse * v + h * f, v);
	}
		

	
	for (auto particle:particles) {
		int index = particle->i;
			
		if (index % cols == 0 || (index + 3) % cols == 0) {
			particle->v = Vector3d(0.0, 0.0, 0.0);
			particle->x += particle->v * h;
			continue;
		}


		if (particle->i >= 0) {
			particle->v = x.segment<3>(particle->i);
			particle->x += particle->v * h;
		}

		if (particle->x.y() < 0) {
			particle->x = Vector3d(particle->x.x(), 0, particle->x.z());
			particle->v = 0.5 * Vector3d(particle->v.x(), abs(particle->v.z()), particle->v.z());
		}
	}
	
	
	// Update position and normal buffers
	updatePosNor();
}

void Cloth::init()
{
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	
	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size()*sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	
	glGenBuffers(1, &texBufID);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glBufferData(GL_ARRAY_BUFFER, texBuf.size()*sizeof(float), &texBuf[0], GL_STATIC_DRAW);
	
	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size()*sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	assert(glGetError() == GL_NO_ERROR);
}

void Cloth::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> p) const
{
	// Draw mesh
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(1.0, 1.0, 0.0).data());
	MV->pushMatrix();
	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void*)0);
	int h_nor = p->getAttribute("aNor");
	glEnableVertexAttribArray(h_nor);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void*)0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	for (int i = 0; i < rows; ++i) {
		glDrawElements(GL_TRIANGLE_STRIP, 2 * cols, GL_UNSIGNED_INT, (const void*)(2 * cols * i * sizeof(unsigned int)));
	}
	glDisableVertexAttribArray(h_nor);
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
