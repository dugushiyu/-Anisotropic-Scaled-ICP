#pragma once
// Dugushiyu.11,2020  updata
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <cmath>
#include <vector>
#define M_PI 3.1415926

struct ASICPparas
{
	double xscalemin;
	double xscalemax;
	double yscalemin;
	double yscalemax;
	double zscalemin;
	double zscalemax;
	double itethreshold;
	double itemax;
	bool uniformityflag;
	bool estimateflag;
};

class ASICP
{
public:
	ASICP();
	~ASICP();

	/////////////////////////////////////////////////////////
	void setParas(ASICPparas paras);
	void estimateScalesFromPoints(Eigen::MatrixXd &p, Eigen::MatrixXd &m,
		Eigen::Matrix3d &Ascale, Eigen::Matrix3d &initR, bool uniformityflag = true);

	int asicp_md(Eigen::MatrixXd &points,
		Eigen::MatrixXd  &model,
		Eigen::Matrix3d &R,
		Eigen::Matrix3d &A,
		Eigen::Vector3d &t,
		double &FRE, double threshold,
		Eigen::VectorXd &FREmag);

	void ASMajor_point_register(Eigen::MatrixXd &XX, Eigen::MatrixXd &YY,
		Eigen::Matrix3d &Q,
		Eigen::Matrix3d &A,
		Eigen::Vector3d &t,
		double &FRE,
		double threshold,
		Eigen::VectorXd &FREMag);

	void ASMajor_point_register(Eigen::MatrixXd &XX, Eigen::MatrixXd &YY,
		Eigen::Matrix3d &Q,
		Eigen::Matrix3d &A,
		Eigen::Vector3d &t,
		double &FRE,
		Eigen::VectorXd &FREMag);

	void q2m3x3(Eigen::VectorXd &qin, Eigen::Matrix3d &m);

	void svdcmp(Eigen::MatrixXd &u, Eigen::Vector3d &w, Eigen::Matrix3d &a, Eigen::Matrix3d &v);

	double MAX(double a, double b);

	double MIN(double a, double b);

	double pythag(double a, double b);

	void FourtyRotations(Eigen::MatrixXd &q);

	double scale_minx;
	double scale_maxx;
	double scale_miny;
	double scale_maxy;
	double scale_minz;
	double scale_maxz;
	double ite_threshold;
	double ite_max;
	bool uniformity_flag;
	bool estimate_flag;
};

