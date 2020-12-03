#include "ASICPv2.h"
#include "nanoflann.hpp"

ASICP::ASICP()
{
	scale_minx = 0.9;
	scale_maxx = 1.1;
	scale_miny = 0.9;
	scale_maxy = 1.1;
	scale_minz = 0.9;
	scale_maxz = 1.1;
	ite_threshold = 1e-5;
	ite_max = 1000;
	uniformity_flag = true;
	estimate_flag = true;
}

ASICP::~ASICP()
{
}

////////////////////////////////////////////////////////////////////////////
#ifndef SIGN
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#endif

void ASICP::setParas(ASICPparas paras)
{
	scale_minx = paras.xscalemin;
	scale_maxx = paras.xscalemax;
	scale_miny = paras.yscalemin;
	scale_maxy = paras.yscalemax;	
	scale_minz = paras.zscalemin;
	scale_maxz = paras.zscalemax;
	ite_threshold = paras.itethreshold;
	uniformity_flag=paras.uniformityflag;
}

void ASICP::estimateScalesFromPoints(Eigen::MatrixXd &p, Eigen::MatrixXd &m,
	Eigen::Matrix3d &Ascale,Eigen::Matrix3d &initR,bool uniformityflag)
{
	int nP = p.cols();
	int nM = m.cols();

	Eigen::MatrixXd l(3, nP);

	// perform the initial rotation
	l = initR * p;

	// find the centroid of both clouds
	Eigen::Vector3d cl = l.rowwise().mean();
	Eigen::Vector3d cr = m.rowwise().mean();

	// translate the input points by their centroids
	Eigen::MatrixXd lprime = l.colwise() - cl;
	Eigen::MatrixXd rprime = m.colwise() - cr;

	// compute the covariance matrix
	Eigen::Matrix3d mu(3, 3), lambda(3, 3);

	// eigen velues are propotional to the number of the points
	mu = 1.0 / (double)nP * lprime * lprime.transpose();
	lambda = 1.0 / (double)nM * rprime * rprime.transpose();

	// solve for eigenvectors and eigenvalues
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
	es.compute(mu);
	Eigen::Matrix3d X_evec = es.eigenvectors();
	Eigen::Vector3d X_eval = es.eigenvalues();
	es.compute(lambda);
	Eigen::Matrix3d Y_evec = es.eigenvectors();
	Eigen::Vector3d Y_eval = es.eigenvalues();
	double minx = X_eval(0)<X_eval(1) ? (X_eval(0)<X_eval(2) ? X_eval(0) : X_eval(2)) : (X_eval(1)<X_eval(2) ? X_eval(1) : X_eval(2));
	double maxx = X_eval(0)<X_eval(1) ? (X_eval(1)<X_eval(2) ? X_eval(2) : X_eval(1)) : (X_eval(0)<X_eval(2) ? X_eval(2) : X_eval(0));
	double midx = X_eval(0) + X_eval(1) + X_eval(2) - minx - maxx;
	double miny = Y_eval(0)<Y_eval(1) ? (Y_eval(0)<Y_eval(2) ? Y_eval(0) : Y_eval(2)) : (Y_eval(1)<Y_eval(2) ? Y_eval(1) : Y_eval(2));
	double maxy = Y_eval(0)<Y_eval(1) ? (Y_eval(1)<Y_eval(2) ? Y_eval(2) : Y_eval(1)) : (Y_eval(0)<Y_eval(2) ? Y_eval(2) : Y_eval(0));
	double midy = Y_eval(0) + Y_eval(1) + Y_eval(2) - miny - maxy;
	double midsum = isnan(midy / midx) ? 1 : midy / midx;
	double minsum = isnan(miny / minx) ? 1 : miny / minx;
	double maxsum = isnan(maxy / maxx) ? 1 : maxy / maxx;


	double initScale = (maxsum + midsum + minsum) / 3.0;
	Ascale = Eigen::Matrix3d::Zero();
	if (uniformityflag)	{
		Ascale(0, 0) = initScale; 
		Ascale(1, 1) = initScale; 
		Ascale(2, 2) = initScale;
	} else {
		Ascale(0, 0) = Y_eval(0) / X_eval(0);
		Ascale(1, 1) = Y_eval(1) / X_eval(1); 
		Ascale(2, 2) = Y_eval(2) / X_eval(2);

		Ascale(0, 0) = isnan(Ascale(0, 0)) ? 0 : Ascale(0, 0);
		Ascale(1, 1) = isnan(Ascale(1, 1)) ? 0 : Ascale(1, 1);
		Ascale(2, 2) = isnan(Ascale(2, 2)) ? 0 : Ascale(2, 2);

	}
	

	std::cout << maxsum << std::endl << midsum << std::endl << minsum << std::endl;
	return;
}

double ASICP::MAX(double a, double b)
{
	double arg1 = a, arg2 = b;

	return ((arg1 > arg2) ? arg1 : arg2);
}

double ASICP::MIN(double a, double b)
{
	double arg1 = a, arg2 = b;

	return ((arg1 < arg2) ? arg1 : arg2);
}

double ASICP::pythag(double a, double b)
{
	// computes (a^2 + b^2)^(1/2) without destructive underflow or verflow
	double absa = fabs(a);
	double absb = fabs(b);

	if (absa > absb)
		return (absa * sqrt(1.0 + (absb / absa)*(absb / absa)));
	else
		return (absb == 0.0 ? 0.0 : absb*sqrt(1.0 + (absa / absb)*(absa / absb)));
}

void ASICP::svdcmp(Eigen::MatrixXd &u, Eigen::Vector3d &w, Eigen::Matrix3d &a, Eigen::Matrix3d &v)
{
	// "u" is the given input matrix
	// "w" is a vector of size n that contains the singular values
	//     of the diagonal matrix
	// 
	// u = a*w*transpose(t);
	//
	size_t m = u.rows();
	size_t n = u.cols();
	int i, j, jj, its, k, l, nm;
	int flag;
	double anorm, c, f, g, h, s, scale, x, y, z;

	if ((v.rows() != m) || (v.cols() != n)) v.resize(m, n);
	if (w.size() != n)  w.resize(n);
						
	Eigen::VectorXd rv1(n);
	a = u;

	g = scale = anorm = 0.0;

	// Householder reduction to bidiagonal form.
	for (i = 0; i < n; i++) {
		l = i + 1;
		rv1(i) = scale * g;
		g = s = scale = 0.0;

		if (i < m) {
			for (k = i; k < m; k++){
				scale += fabs(a(k, i));
			}

			if (scale) {
				for (k = i; k < m; k++) {
					a(k, i) /= scale;
					s += a(k, i) * a(k, i);
				}

				f = a(i, i);
				g = -SIGN(sqrt(s), f);

				h = f * g - s;
				a(i, i) = f - g;
				for (j = l; j < n; j++) {
					for (s = 0.0, k = i; k < m; k++) { 
						s += a(k, i) * a(k, j); 
					}
					f = s / h;
					for (k = i; k < m; k++) a(k, j) += f * a(k, i);
				}
				for (k = i; k < m; k++) { 
					a(k, i) *= scale;
				}
			}
		}

		w(i) = scale * g;
		g = s = scale = 0.0;

		if ((i < m) && (i != (n-1))) {
			for (k = l; k < n; k++) {
				scale += fabs(a(i, k));
			}
			if (scale) {
				for (k = l; k < n; k++) {
					a(i, k) /= scale;
					s += a(i, k) * a(i, k);
				}

				f = a(i, l);
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a(i, l) = f - g;
				for (k = l; k < n; k++) rv1(k) = a(i, k) / h;
				for (j = l; j < m; j++) {
					for (s = 0.0, k = l; k < n; k++) {
						s += a(j, k) * a(i, k);
					}
					for (k = l; k < n; k++) {
						a(j, k) += s * rv1(k);
					}
				}
				for (k = l; k < n; k++) {
					a(i, k) *= scale;
				}
			}
		}
		anorm = MAX(anorm, (fabs(w(i)) + fabs(rv1(i))));
	}

	// Accumulation of right-hand transformations
	for (i = n-1; i >= 0; i--) {
		if (i < n) {
			if (g) {
				for (j = l; j < n; j++) { // double division to avoid possible underflow
					v(j, i) = (a(i, j) / a(i, l)) / g;
				}

				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < n; k++) {
						s += a(i, k) * v(k, j);
					}
					for (k = l; k < n; k++) {
						v(k, j) += s*v(k, i);
					}
				}
			}
			for (j = l; j < n; j++) {
				v(i, j) = v(j, i) = 0.0;
			}
		}
		v(i, i) = 1.0;
		g = rv1(i);
		l = i;
	}

	for (i = MIN(m, n)-1; i >= 0; i--) {
		// Accumulation of left-hand transformations
		l = i + 1;
		g = w(i);
		for (j = l; j < n; j++) a(i, j) = 0.0;
		if (g) {
			g = 1.0 / g;
			for (j = l; j < n; j++) {
				for (s = 0.0, k = l; k < m; k++) {
					s += a(k, i) * a(k, j);
				}
				f = (s / a(i, i)) * g;
				for (k = i; k < m; k++) {
					a(k, j) += f * a(k, i);
				}
			}
			for (j = i; j < m; j++) {
				a(j, i) *= g;
			}
		}
		else for (j = i; j < m; j++) {
			a(j, i) = 0.0;
		}
		++a(i, i);
	}

	// Diagonalization of the bidiagonal form:
	// Loop over singular values, and over allowed iterations
	for (k = n-1; k >= 0; k--) {
		for (its = 1; its <= 30; its++) {
			flag = 1;
			for (l = k; l >= 0; l--) {
				// test for splitting
				// note that rv1(1) is always zero
				nm = l - 1;
				if (fabs((double)(fabs(rv1(l)) + anorm) - anorm)<1e-3) {
					flag = 0;
					break;
				}
				if (fabs((double)(fabs(w(nm)) + anorm) - anorm)<1e-3) break;
			}

			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) {
					f = s*rv1(i);
					rv1(i) = c * rv1(i);

					if (fabs((double)(fabs(f) + anorm) - anorm)<1e-3) break;
					g = w(i);
					h = pythag(f, g);

					w(i) = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;

					for (j = 0; j < m; j++) {
						y = a(j, nm);
						z = a(j, i);
						a(j, nm) = y * c + z * s;
						a(j, i) = z * c - y * s;
					}
				}
			}

			z = w(k);
			if (l == k) {
				// Convergence
				// singular value is made nonnegative
				if (z < 0.0) {
					w(k) = -z;
					for (j = 0; j < n; j++) {
						v(j, k) = -v(j, k);
					}
				}
				break;
			}

			assert(its != 30); // check this

			x = w(l);
			nm = k - 1;
			y = w(nm);
			g = rv1(nm);
			h = rv1(k);
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = pythag(f, 1.0);
			f = ((x - z) * (x + z) + h*((y / (f + SIGN(g, f))) - h)) / x;
			c = s = 1.0;

			// Next QR transformation
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1(i);
				y = w(i);
				h = s * g;
				g = c * g;
				z = pythag(f, h);
				rv1(j) = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;
				for (jj = 0; jj < n; jj++) {
					x = v(jj, j);
					z = v(jj, i);
					v(jj, j) = x * c + z * s;
					v(jj, i) = z * c - x * s;
				}
				z = pythag(f, h);
				w(j) = z;  // Rotation can be arbitrary if z = 0
				if (z) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;

				for (jj = 0; jj < m; jj++) {
					y = a(jj, j);
					z = a(jj, i);
					a(jj, j) = y * c + z * s;
					a(jj, i) = z * c - y * s;
				}
			}
			rv1(l) = 0.0;
			rv1(k) = f;
			w(k) = x;
		}
	}
}

// generate 40 rotations that includes the tetrahedral and
// the octahedral/hexahedral group as per the original ICP paper
// by Besl and McKay (page 247)
// q is a 4x40 quaterion matrix where each column of q is an unit quaternion
void ASICP::FourtyRotations(Eigen::MatrixXd &q)
{
	Eigen::MatrixXd tempq(4, 54); // Matrix<double> tempq(4, 54);

	Eigen::VectorXd q0(2); q0(0) = 1; q0(1) = 0;
	Eigen::VectorXd q1(3); q1(0) = 1; q1(1) = 0; q1(2) = -1;
	Eigen::VectorXd q2(3); q2(0) = 1; q2(1) = 0; q2(2) = -1;
	Eigen::VectorXd q3(3); q3(0) = 1; q3(1) = 0; q3(2) = -1;

	int counter = 0;
	Eigen::MatrixXd  temp(4, 54);
	double ll = 1.0;
	for (int i = 0; i < q0.size(); i++)
		for (int j = 0; j < q1.size(); j++)
			for (int k = 0; k < q2.size(); k++)
				for (int l = 0; l < q3.size(); l++)
				{
					// scalar component of the quaternion
					tempq(3, counter) = ll * q0(i); 
					tempq(0, counter) = ll * q1(j);
					tempq(1, counter) = ll * q2(k);
					tempq(2, counter) = ll * q3(l);
					counter++;
				}


	// 41st quaternion is not valid
	// 42nd t0 54th are duplicates
	q.resize(4, 40); 
	for (int i = 0; i < 40; i++)
		for (int j = 0; j < 4; j++)
		{
			q(j, i) = tempq(j, i);
		}
}

// convert a quaternion to 3x3 rotation matrix
//
// a quaternion is defined as:
//
// q[0] = v1*sin(phi/2)
// q[1] = v2*sin(phi/2)
// q[2] = v3*sin(phi/2)
// q[3] =    cos(phi/2)

void ASICP::q2m3x3(Eigen::VectorXd &qin, Eigen::Matrix3d &m)
{
	int nq = qin.size();
	if (nq != 4)
	{
		std::cerr << "Expecting the quternion to be a 4x1 vector" << std::endl;
		exit(1);
	}

	// normalize the quaternion
	double l = sqrt(qin[0] * qin[0] +
		qin[1] * qin[1] +
		qin[2] * qin[2] +
		qin[3] * qin[3]);

	Eigen::VectorXd q = 1.0 / l * qin;

	m.resize(3, 3);
	double xx = q[0] * q[0];
	double yy = q[1] * q[1];
	double zz = q[2] * q[2];

	double xy = q[0] * q[1];
	double xz = q[0] * q[2];

	double yz = q[1] * q[2];

	double wx = q[3] * q[0];
	double wy = q[3] * q[1];
	double wz = q[3] * q[2];


	m(0, 0) = 1. - 2. * (yy + zz);
	m(0, 1) = 2. * (xy - wz);
	m(0, 2) = 2. * (xz + wy);

	m(1, 0) = 2. * (xy + wz);
	m(1, 1) = 1. - 2. * (xx + zz);
	m(1, 2) = 2. * (yz - wx);

	m(2, 0) = 2. * (xz - wy);
	m(2, 1) = 2. * (yz + wx);
	m(2, 2) = 1. - 2. * (xx + yy);
	return;
}


void ASICP::ASMajor_point_register(Eigen::MatrixXd &XX, Eigen::MatrixXd &YY,
	Eigen::Matrix3d &Q,
	Eigen::Matrix3d &A,
	Eigen::Vector3d &t,
	double &FRE,
	Eigen::VectorXd &FREMag)
{
	double threshold = ite_threshold;
	ASMajor_point_register(XX, YY, Q, A, t, FRE, threshold, FREMag);
}


void ASICP::ASMajor_point_register(Eigen::MatrixXd &XX, Eigen::MatrixXd &YY,
	Eigen::Matrix3d &Q,
	Eigen::Matrix3d &A,
	Eigen::Vector3d &t,
	double &FRE,
	double threshold,
	Eigen::VectorXd &FREMag)
{
	// double check the dimensions
	int XXx = XX.rows();
	int XXy = XX.cols();
	int YYx = YY.rows();
	int YYy = YY.cols();

	if (XXx != 3 || YYx != 3)
	{
		std::cerr << "inputs must use column vectors" << std::endl;
		exit(1);
	}
	if (XXy != YYy)
	{
		std::cerr << "input point clouds must contain the same number of points" << std::endl;
		exit(1);
	}

	// for simplicity, we'll follow the notation of Dossee and Berge to use
	// row-vectors instead of column vectors.
	Eigen::MatrixXd X = XX.transpose();
	Eigen::MatrixXd Y = YY.transpose();
	
	Eigen::MatrixXd Xbar = X.colwise().mean();
	Eigen::MatrixXd Ybar = Y.colwise().mean();

	Eigen::MatrixXd Xtilde(XXy,XXx), Ytilde(YYy,YYx);
	for (int i = 0; i < XXy; i++)
	{
		Xtilde.row(i) = X.row(i) - Xbar.row(0);
		Ytilde.row(i) = Y.row(i) - Ybar.row(0);
	}

	// normlize Xtilde (page 115 of the paper)
	Eigen::Vector3d S = Eigen::Vector3d::Zero();
	for (int i = 0; i < X.rows(); i++)
	{
		for (int j = 0; j < X.cols(); j++)
		{
			S(j) += (Xtilde(i, j) * Xtilde(i, j));
		}
	}

	S(0) = S(0)<1e-9 ? 1 : sqrt(S(0));
	S(1) = S(1)<1e-9 ? 1 : sqrt(S(1));
	S(2) = S(2)<1e-9 ? 1 : sqrt(S(2));

	for (int i = 0; i < X.rows(); i++)
	{
		for (int j = 0; j < X.cols(); j++)
		{
			Xtilde(i, j) /= S(j);
		}
	}

	// compute the cross correlation matrix
	Eigen::MatrixXd B = Ytilde.transpose() * Xtilde;

	// compute the intial rotation
	Eigen::Matrix3d U, V;

	// Matrix<double> U, V;
	svdcmp(B, S, U, V);

	Eigen::Matrix3d UV = U * V;
	Eigen::Matrix3d dd = Eigen::Matrix3d::Identity();
	dd(2, 2) = (UV).determinant();
	Q = U * dd * V.transpose();

	// find the residual
	Eigen::MatrixXd  FREvect = Xtilde * Q.transpose() - Ytilde;

	FRE = 0.0;
	for (int i = 0; i < FREvect.rows(); i++)
	{
		FRE += (FREvect(i, 0) * FREvect(i, 0) +
			FREvect(i, 1) * FREvect(i, 1) +
			FREvect(i, 2) * FREvect(i, 2));

	}
	FRE = sqrt(FRE / (double)FREvect.rows());

	double FRE_orig = 2.0 * (FRE + threshold);
	int maxiter = 1000;
	int itenum = 0;
	Eigen::Matrix3d QB(3, 3);
	Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
	while (fabs(FRE_orig - FRE) > threshold && itenum<maxiter)
	{
		QB = Q.transpose() * B;
		I(0, 0) = QB(0, 0);
		I(1, 1) = QB(1, 1);
		I(2, 2) = QB(2, 2);

		Eigen::MatrixXd temBI = B*I;

		svdcmp(temBI, S, U, V);
		UV = U * V;

		dd(2, 2) = (UV).determinant();
		Q = U * dd * V.transpose();

		// calculate the residual
		FREvect = Xtilde * Q.transpose() - Ytilde;
		FRE_orig = FRE;
		FRE = 0.0;
		for (int i = 0; i < FREvect.rows(); i++)
		{
			FRE += (FREvect(i, 0) * FREvect(i, 0) +
				FREvect(i, 1) * FREvect(i, 1) +
				FREvect(i, 2) * FREvect(i, 2));
		}
		FRE = sqrt(FRE / (double)FREvect.rows());
		itenum++;
	}

	for (size_t i = 0; i < XXy; i++)
	{
		Xtilde.row(i) = X.row(i) - Xbar.row(0);
	}
	B = Ytilde.transpose() * Xtilde;

	// A = diag( diag( B'*Q ) ./ diag( Xtilde' * Xtilde ) );
	// reuse U and V
	U = B.transpose() * Q;
	V = Xtilde.transpose() * Xtilde;
	A = Eigen::Matrix3d::Zero();
	
	A(0, 0) = U(0, 0) / V(0, 0);
	A(1, 1) = U(1, 1) / V(1, 1);
	A(2, 2) = U(2, 2) / V(2, 2);

	A(0, 0) = isnan(A(0, 0)) ? 1.0 : A(0, 0);
	A(1, 1) = isnan(A(1, 1)) ? 1.0 : A(1, 1);
	A(2, 2) = isnan(A(2, 2)) ? 1.0 : A(2, 2);

	// now calculate the translation
	t[0] = t[1] = t[2] = 0.0;
	/*
	U = YY - Q * A * XX; // reuse U

	for (int i = 0; i < U.cols(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			t[j] += U(j, i);
		}
	}
	for (int i = 0; i < 3; i++)
	{
		t[i] /= (double)U.cols();
	}
	*/
	// calculate the FRE and the translation
	FREvect = Ytilde - Xtilde * A * Q.transpose();

	FRE = 0.0;

	if (FREMag.size() != FREvect.rows())
	{
		FREMag.resize(FREvect.rows());
	}

	for (int i = 0; i < FREvect.rows(); i++)
	{
		FREMag(i) = sqrt(FREvect(i, 0) * FREvect(i, 0) +
			FREvect(i, 1) * FREvect(i, 1) +
			FREvect(i, 2) * FREvect(i, 2));
		FRE += FREMag[i];
	}
	FRE = (FRE / (double)FREvect.rows());

	t = Ybar.transpose() - Q * A * Xbar.transpose();
}


int ASICP::asicp_md(Eigen::MatrixXd &points,
	Eigen::MatrixXd  &model,
	Eigen::Matrix3d &R,
	Eigen::Matrix3d &A,
	Eigen::Vector3d &t,
	double &FRE, double threshold,
	Eigen::VectorXd &FREmag)
{
	if (points.rows() != 3 || model.rows() != 3) {
		std::cerr << "X and Y must be column matrices with 3 rows" << std::endl;
		return -1;
	}

	int nPoints = points.cols();
	int nModel = model.cols();

	Eigen::MatrixXd r(3, nPoints), l(3, nPoints), lprime(3, nPoints), query_pt(3, nPoints);
	Eigen::MatrixXd query_ptE(3, nPoints), data_ptsE(3, nModel);
	Eigen::MatrixXd data_ptsEClose(3, nPoints);

	// the ICP loop
	Eigen::MatrixXd residual;
	std::vector<double>  residMag;

	double oldFRE = FRE;
	int maxIter = ite_max;
	int nIter = 0;

	// estimate the scales based on the eigenvalues of the covariance matrices
	lprime = R * points;
	lprime = lprime.colwise() + t;
	Eigen::Matrix3d Ascale;
	if (estimate_flag) {
		estimateScalesFromPoints(lprime, model, Ascale, R, uniformity_flag);
		A = Ascale;
	} else {
		 Ascale = A;
	}
	
	std::cerr << std::endl << "Initial Scales: " << std::endl << A << std::endl;

	Eigen::Matrix3d RR, AA;
	Eigen::Vector3d tt;
	double residuals;
	bool changedScales;

	lprime = R * A * points;

	//initialise kd-tree
	typedef nanoflann::KDTreeEigenMatrixAdaptor<
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
		3, nanoflann::metric_L2, false>
		kd_tree_t;

	Eigen::MatrixXd cloud_tgt(3, nPoints);
	Eigen::MatrixXd cloud_src(3, nPoints);
	while (nIter < maxIter)
	{
		std::cerr << std::endl << "i: " << nIter << std::endl;
		// loop for a fixed number of times 'cus we may never converge
		// in this loop, "l" is the transformed points,
		// "r" is the nearest neighbour in the model
		l = R * A * points;
		l = l.colwise() + t;

		// estimate the scales needed to be used in Mahalanobis distance
		// what we do here is to use the correspondanding fiducials from
		// the previous iteration of ICP and register that to the current
		// iteration to get an idea of the scaling.

		ASMajor_point_register(lprime, l, RR, AA, tt, residuals, threshold, FREmag);
		lprime = l; 

		Eigen::MatrixXd sqrtAA;
		sqrtAA = AA;
		sqrtAA(0, 0) = isnan(sqrt(AA(0, 0))) ? 1 : sqrt(AA(0, 0));
		sqrtAA(1, 1) = isnan(sqrt(AA(1, 1))) ? 1 : sqrt(AA(1, 1));
		sqrtAA(2, 2) = isnan(sqrt(AA(2, 2))) ? 1 : sqrt(AA(2, 2));

		  // copy all the model points into ANN structure
		  // NOTE: since we are interested in Mahalabonis Distance,
		  // both the data/model points needs to be scaled
		for (int i = 0; i < nModel; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				data_ptsE(j, i) = model(j, i) / sqrtAA(j, j);
			} // j
		} // i

	//////////////////////////////////////////////////////////////////////////
		kd_tree_t kd_tree(3, data_ptsE, 100);
		kd_tree.index->buildIndex();

		// find the nearest neighbour of each point in 'l'
		int colnum = 0;
		for (int i = 0; i < nPoints; i++)
		{
			size_t ret_index;
			double out_dist_sqr;
			nanoflann::KNNResultSet<double> result_set(1);
			result_set.init(&ret_index, &out_dist_sqr);
			double query[3];
			query[0] = l(0, i) / sqrtAA(0, 0);
			query[1] = l(1, i) / sqrtAA(1, 1);
			query[2] = l(2, i) / sqrtAA(2, 2);

			// find closest points in the scaled Y points to the current transformed points of X
			bool findflag = kd_tree.index->findNeighbors(result_set,
				&query[0],nanoflann::SearchParams(100));
			if (findflag)
			{
				data_ptsEClose.col(colnum) = data_ptsE.col((int)ret_index);
				for (int j = 0; j < 3; j++)
				{
					cloud_tgt(j, i) = data_ptsEClose(j, colnum) * sqrtAA(j, j);
				}
				colnum++;
			}
			else
			{
				std::cerr << "no find" << std::endl;
			}
		
		} // i	
		  ///////////////////////////////////////////////////////////////////////////////////
		  // once the correspondances are found, solve for the
		  // anisotropic-scaled orthogonal procrustes analysis
		  // using the solution by Dosse and Ten Berge
		if (colnum>0)
		{
			data_ptsEClose.resize(3, colnum);
			ASMajor_point_register(points, cloud_tgt,R, A, t,FRE, threshold, FREmag);
			changedScales = false;
		}
		else {
			changedScales = true;
		}
		std::cerr << "Scales: " << std::endl<<A<<std::endl;

		// need to regularize scaling factors A here, and
		// recompute FRE if necessary.
		// In practice, scaling factors need to be bounded, otherwise
		// trivial (and wrong) solution such as scaling~=0 would occure
		// and the computed FRE would be minimal.

		if (A(0, 0) <  scale_minx * Ascale(0,0))
		{
			A(0, 0) = scale_minx * Ascale(0, 0);
			changedScales = true;
		}
		if (A(0, 0) > scale_maxx * Ascale(0, 0))
		{
			A(0, 0) = scale_maxx * Ascale(0, 0);
			changedScales = true;
		}
		if (A(1, 1) <  scale_miny * Ascale(1, 1))
		{
			A(1, 1) = scale_miny * Ascale(1, 1);
			changedScales = true;
		}
		if (A(1, 1) > scale_maxy * Ascale(1, 1))
		{
			A(1, 1) = scale_maxy * Ascale(1, 1);
			changedScales = true;
		}
		if (A(2, 2) <  scale_minz * Ascale(2, 2))
		{
			A(2, 2) = scale_minz * Ascale(2, 2);
			changedScales = true;
		}
		if (A(2, 2) > scale_maxz * Ascale(2, 2))
		{
			A(2, 2) = scale_maxz * Ascale(2, 2);
			changedScales = true;
		}

		// scales has been changed, so re-compute the rotation and FRE
		if (changedScales)
		{
			//std::cerr << "Scales changed" << std::endl;
			l = A * points;
			Eigen::Matrix<double, 4, 4> dTransRT;

			for (size_t i = 0; i < nPoints; i++)
			{
				cloud_src(0, i) = l(0, i);
				cloud_src(1, i) = l(1, i);
				cloud_src(2, i) = l(2, i);
			}
			dTransRT = Eigen::umeyama(cloud_src, cloud_tgt);
			R = dTransRT.block(0,0,3,3);
			t[0] = dTransRT(0, 3);	  
			t[1] = dTransRT(1, 3);	 
			t[2] = dTransRT(2, 3);

			//
			Eigen::MatrixXd  FREvectT = ((R*cloud_src).colwise() + t);
			Eigen::MatrixXd  FREvect =( FREvectT  - cloud_tgt ) ;
			
			for (int i = 0; i < FREvect.rows(); i++)
			{
				FRE += (FREvect(i, 0) * FREvect(i, 0) +
					FREvect(i, 1) * FREvect(i, 1) +
					FREvect(i, 2) * FREvect(i, 2));
			}
			FRE = sqrt(FRE / (double)FREvect.rows());

		}

		if (oldFRE == FRE)
		{
			nIter = maxIter + 1;
		}

		std::cerr << std::endl << "FRE= " << FRE << std::endl;

		oldFRE = FRE;
		if (FRE <= threshold)
		{
			nIter = maxIter + 1;
		}
		nIter++;

	} // while

	return 1;
}
