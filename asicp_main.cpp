/*=========================================================================

Program:   Robarts ICP
Module:    $RCSfile: asicp_main.cpp,v $
Creator:   Elvis C. S. Chen <chene@robarts.ca>
Language:  C++
Author:    $Author: Elvis Chen $
Date:      $Date: 2014/03/04 12:49:30 $
Version:   $Revision: 0.99 $

==========================================================================

This is an open-source copyright license based on BSD 2-Clause License,
see http://opensource.org/licenses/BSD-2-Clause

Copyright (c) 2013, Elvis Chia-Shin Chen
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/

// C++ includes
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>

// local includes
#include "matrix.h"
#include "mathUtils.h"
#include "asicp.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "ASICPv2.h"
#include "ASICPv3.h"

#define M_PI 3.1415926
int main2(  )
  {
  
  std::cerr << std::endl
    << "The Iterative Closest Point with Anisotropic Scaling (ASICP)" << std::endl
    << "using Mahalanobis Distance" << std::endl << std::endl;

  size_t nModels = 29;

  Eigen::Matrix3d R2 = /*Eigen::Matrix3d::Identity();*/ (Eigen::AngleAxisd(M_PI / 3.19, Eigen::Vector3d::UnitX()).matrix() *
												   Eigen::AngleAxisd(3.99*M_PI / 2.8, Eigen::Vector3d::UnitY()).matrix() *
												   Eigen::AngleAxisd(5.12*M_PI / 0.4, Eigen::Vector3d::UnitZ()).matrix());
  Eigen::Matrix3d S = Eigen::Matrix3d::Identity(); // Eigen::Vector3d::Random(3).asDiagonal() * 22.2;
  S(0, 0) = 0.92;	S(1, 1) = 0.95;	S(2, 2) = 0.98;

  Eigen::Vector3d l = Eigen::RowVector3d::Random(3) * 10.7;

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(3, nModels) * 102.4;
  Eigen::MatrixXd Y = (R2 * (S * X)).colwise() + l;

  Matrix<double> models( 3, nModels ), points( 3, nModels);
  for (size_t i = 0; i < nModels; i++)
  {
	  points[0][i] = X(0,i); points[1][i] = X(1, i); points[2][i] = X(2, i);
	  models[0][i] = Y(0, i); models[1][i] = Y(1, i); models[2][i] = Y(2, i);

  }
 
  std::cerr << "Finished reading files" << std::endl;

  // initial rotation
  double FRE = 0.0, tau = 1e-9;
  Matrix<double> A( 3, 3, 0.0 ); // scaling
  Matrix<double> R( 3, 3 );      // rotation
  Vec<double> t(3);              // translation

  Matrix<double> initialQuaternions; // rotation group
  FourtyRotations( initialQuaternions ); // fourty uniformally sampled rotations

  // loop through all the rotation group
  Vec<double> quat(4), minT(3);
  double minRMS = 0.0;
  Matrix<double> minA(3,3), minR(3,3);
  Vec<double> FREMag, minFREMag;

    {
    // go through the rotation group
    for ( int i = 0; i < initialQuaternions.num_cols(); i++ )
      {
      // go through all the rotations
      quat[0] = initialQuaternions[0][i];
      quat[1] = initialQuaternions[1][i];
      quat[2] = initialQuaternions[2][i];
      quat[3] = initialQuaternions[3][i];
      t[0] = t[1] = t[2] = 0.0;
      q2m3x3( quat, R ); // initial guess on rotation
      A = eye( 3, 1.0 ); // initial guess on scaling
      // translation does not matter much


      asicp_md( points, models, R, A, t, FRE, tau, FREMag );


      std::cerr << i << " FRE: " << minRMS << std::endl
        << R << A << t << std::endl;
      if ( i == 0 )
        { 
        minRMS = FRE;
        minR = R;
        minA = A;
        minT = t;
        minFREMag = FREMag;
        }

      if ( FRE < minRMS )
        {
        minRMS = FRE;
        minR = R;
        minA = A;
        minT = t;
        minFREMag = FREMag;
        }
      }

    }
  


  std::cerr << "Final answer: " << minR << minA << minT << std::endl;


  return(0);
  }


  ///////////////////////////////////////////////////////////////
  int mainASICP()
  {

	  std::cerr << std::endl
		  << "The Iterative Closest Point with Anisotropic Scaling (ASICP)" << std::endl
		  << "using Mahalanobis Distance" << std::endl << std::endl;

	  size_t nModels = 200;

	  Eigen::Matrix3d R2 = /*Eigen::Matrix3d::Identity();*/ (Eigen::AngleAxisd(0*M_PI / 3.19, Eigen::Vector3d::UnitX()).matrix() *
		  Eigen::AngleAxisd(0*3.99*M_PI / 2.8, Eigen::Vector3d::UnitY()).matrix() *
		  Eigen::AngleAxisd(5.12*M_PI / 0.4, Eigen::Vector3d::UnitZ()).matrix());
	  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();// Eigen::Vector3d::Random(3).asDiagonal() * 10;
	  S(0, 0) = 0.9;	S(1, 1) = 0.95;	S(2, 2) = 1;

	  /////////////////////////////////////////////////////////////////////////
	 
	  Eigen::Vector3d l = Eigen::RowVector3d::Random(3) * 0;
	  std::cout <<"r2"<<std::endl<< R2 << std::endl << "s"<<std::endl<<S << std::endl <<"L"<<std::endl<< l << std::endl;
	  std::cerr << "============================================================================ " << std::endl;

	  Eigen::MatrixXd XX = Eigen::MatrixXd::Random(3, 200) * 100;
	  Eigen::MatrixXd YY = (R2 * (S * XX)).colwise() + l;
	  
	  Matrix<double> models(3, nModels), points(3, nModels);
	  Eigen::MatrixXd X(3, 200), Y(3, 200);
	  for (int i = 0; i < nModels; i++)
	  {
		  if (1)
		  {
			  points[0][i] = XX(0, i); points[1][i] = XX(1, i); points[2][i] = XX(2, i);
			  X(0,i) = XX(0, i); X(1, i) = XX(1, i); X(2, i) = XX(2, i);
		  }
		  if (1)
		  {
			  models[0][i] = YY(0, i); models[1][i] = YY(1, i); models[2][i] = YY(2, i);
			  Y(0, i) = YY(0, i); Y(1, i) = YY(1, i); Y(2, i) =YY(2, i);

		  }

	  }
	  /*
	  // initial rotation
	  double FRE = 0.0, tau = 1e-9;
	  Matrix<double> A(3, 3, 0.0); // scaling
	  Matrix<double> R(3, 3);      // rotation
	  Vec<double> t(3);              // translation

	  Matrix<double> initialQuaternions; // rotation group
	  FourtyRotations(initialQuaternions); // fourty uniformally sampled rotations

										   // loop through all the rotation group
	  Vec<double> quat(4), minT(3);
	  double minRMS = 0.0;
	  Matrix<double> minA(3, 3), minR(3, 3);
	  Vec<double> FREMag, minFREMag;

	// go through the rotation group
	for (int i = 0; i < initialQuaternions.num_cols(); i++)
	{
		// go through all the rotations
		quat[0] = initialQuaternions[0][i];
		quat[1] = initialQuaternions[1][i];
		quat[2] = initialQuaternions[2][i];
		quat[3] = initialQuaternions[3][i];
		t[0] = t[1] = t[2] = 0.0;
		q2m3x3(quat, R); // initial guess on rotation
		A = eye(3, 1.0); // initial guess on scaling
						// translation does not matter much

		asicp_md(points, models, R, A, t, FRE, tau, FREMag);


		//std::cerr << i << " FRE: " << minRMS << std::endl << R << A << t << std::endl;
		if (i == 0)
		{
			minRMS = FRE;
			minR = R;
			minA = A;
			minT = t;
			minFREMag = FREMag;
		}

		if (FRE < minRMS)
		{
			minRMS = FRE;
			minR = R;
			minA = A;
			minT = t;
			minFREMag = FREMag;
		}
	}

	  std::cerr <<std::endl<< "Final answer: " << minR << minA << minT << std::endl;
	  */
	  std::cerr << "============================================================================ " <<std::endl;
	  
	  Eigen::VectorXd quat3(4);
	  Eigen::Matrix3d R3, minR3;
	  Eigen::Matrix3d A3, minA3;
	  Eigen::Vector3d t3, minT3;
	  double FRE3 = 0, threshold3 = 1e-9;
	  Eigen::VectorXd FREmag3, minFREMag3;
	  ASICP asicpT;
	  ASICPparas paras;
	  paras.xscalemin=0.8;
	  paras.xscalemax=1.2;
	  paras.yscalemin=0.8;
	  paras.yscalemax=1.2;
	  paras.zscalemin=0.8;
	  paras.zscalemax=1.2;
	  paras.itethreshold=1e-9;
	  paras.uniformityflag = true;
	  paras.estimateflag = false;

	  asicpT.setParas(paras);
	  Eigen::MatrixXd initialQuaternions3;
	  asicpT.FourtyRotations(initialQuaternions3);

	  double minRMS3 = 0.0;
	  for (int i = 0; i < initialQuaternions3.cols(); i++)
	  {
		  // go through all the rotations
		  quat3[0] = initialQuaternions3(0, i);
		  quat3[1] = initialQuaternions3(1, i);
		  quat3[2] = initialQuaternions3(2, i);
		  quat3[3] = initialQuaternions3(3, i);
		  t3[0] = t3[1] = t3[2] = 0.0;
		  asicpT.q2m3x3(quat3, R3); 
		  A3 = Eigen::Matrix3d::Identity(); 
		 // translation does not matter much
		  asicpT.asicp_md(X, Y, R3, A3, t3, FRE3, threshold3, FREmag3);
		  if (i == 0)
		  {
			  minRMS3 = FRE3;
			  minR3 = R3;
			  minA3 = A3;
			  minT3 = t3;
			  minFREMag3 = FREmag3;
		  }

		  if (FRE3 < minRMS3)
		  {
			  minRMS3 = FRE3;
			  minR3 = R3;
			  minA3 = A3;
			  minT3 = t3;
			  minFREMag3 = FREmag3;
		  }
	  }

	  std::cerr << "============================================================================ " << std::endl;
	  std::cerr <<std::endl<< "Final answer: " <<std::endl<< minR3 << std::endl << minA3 << std::endl<< minT3 << std::endl;

	  std::cerr << "============================================================================ " << std::endl;
	  std::cout << "r2" << std::endl << R2 << std::endl << "s" << std::endl << S << std::endl << "L" << std::endl << l << std::endl;

	  return(0);
  }

  ///////////////////////////////////////////////////////////////
  int main()
  {

	  std::cerr << std::endl
		  << "The Iterative Closest Point with Anisotropic Scaling (ASICP)" << std::endl
		  << "using Mahalanobis Distance" << std::endl << std::endl;

	  size_t nModels = 200;

	  Eigen::Matrix2d R2 = Eigen::Matrix2d::Identity();// (Eigen::AngleAxisd(0 * M_PI / 3.19, Eigen::Vector2d::UnitX()).matrix() *
		 // Eigen::AngleAxisd(0 * 3.99*M_PI / 2.8, Eigen::Vector2d::UnitY()).matrix() *
		//  Eigen::AngleAxisd(5.12*M_PI / 0.4, Eigen::Vector2d::UnitZ()).matrix()
		//	  );
	  Eigen::Matrix2d S = Eigen::Matrix2d::Identity();// Eigen::Vector3d::Random(3).asDiagonal() * 10;
	  S(0, 0) = 0.9;	S(1, 1) = 0.95;//	S(2, 2) = 1;

	  /////////////////////////////////////////////////////////////////////////

	  Eigen::Vector2d l = Eigen::RowVector2d::Random(2) * 0;
	  std::cout << "r2" << std::endl << R2 << std::endl << "s" << std::endl << S << std::endl << "L" << std::endl << l << std::endl;
	  std::cerr << "============================================================================ " << std::endl;

	  Eigen::MatrixXd XX = Eigen::MatrixXd::Random(2, 200) * 100;
	  Eigen::MatrixXd YY = (R2 * (S * XX)).colwise() + l;

	  Matrix<double> models(2, nModels), points(2, nModels);
	  Eigen::MatrixXd X(2, 200), Y(2, 200);
	  for (int i = 0; i < nModels; i++)
	  {
		  if (1)
		  {
			  points[0][i] = XX(0, i); points[1][i] = XX(1, i); //points[2][i] = XX(2, i);
			  X(0, i) = XX(0, i); X(1, i) = XX(1, i);// X(2, i) = XX(2, i);
		  }
		  if (1)
		  {
			  models[0][i] = YY(0, i); models[1][i] = YY(1, i); //models[2][i] = YY(2, i);
			  Y(0, i) = YY(0, i); Y(1, i) = YY(1, i);// Y(2, i) = YY(2, i);
		  }

	  }
	
	  std::cerr << "============================================================================ " << std::endl;

	  Eigen::VectorXd quat3(3);
	  Eigen::Matrix2d R3, minR3;
	  Eigen::Matrix2d A3, minA3;
	  Eigen::Vector2d t3, minT3;
	  double FRE3 = 0, threshold3 = 1e-9;
	  Eigen::VectorXd FREmag3, minFREMag3;
	  ASICP2 asicpT;
	  ASICPparas2 paras;
	  paras.xscalemin = 0.8;
	  paras.xscalemax = 1.2;
	  paras.yscalemin = 0.8;
	  paras.yscalemax = 1.2;
	  paras.zscalemin = 0.8;
	  paras.zscalemax = 1.2;
	  paras.itethreshold = 1e-9;
	  paras.uniformityflag = true;
	  paras.estimateflag = false;

	  asicpT.setParas(paras);
	  Eigen::MatrixXd initialQuaternions3;
	  asicpT.FourtyRotations(initialQuaternions3);

	  double minRMS3 = 0.0;
	  for (int i = 0; i < initialQuaternions3.cols(); i++)
	  {
		  // go through all the rotations
		  //quat3[0] = initialQuaternions3(0, i);
		  //quat3[1] = initialQuaternions3(1, i);
		  //quat3[2] = initialQuaternions3(2, i);
		  //quat3[3] = initialQuaternions3(3, i);
		  //t3[0] = t3[1] = t3[2] = 0.0;
		  //asicpT.q2m3x3(quat3, R3);
		  R3 = Eigen::Matrix2d::Identity();
		  A3 = Eigen::Matrix2d::Identity();
		  // translation does not matter much
		  asicpT.asicp_md(X, Y, R3, A3, t3, FRE3, threshold3, FREmag3);
		  if (i == 0)
		  {
			  minRMS3 = FRE3;
			  minR3 = R3;
			  minA3 = A3;
			  minT3 = t3;
			  minFREMag3 = FREmag3;
		  }

		  if (FRE3 < minRMS3)
		  {
			  minRMS3 = FRE3;
			  minR3 = R3;
			  minA3 = A3;
			  minT3 = t3;
			  minFREMag3 = FREmag3;
		  }
	  }

	  std::cerr << "============================================================================ " << std::endl;
	  std::cerr << std::endl << "Final answer: " << std::endl << minR3 << std::endl << minA3 << std::endl << minT3 << std::endl;

	  std::cerr << "============================================================================ " << std::endl;
	  std::cout << "r2" << std::endl << R2 << std::endl << "s" << std::endl << S << std::endl << "L" << std::endl << l << std::endl;

	  return(0);
  }
