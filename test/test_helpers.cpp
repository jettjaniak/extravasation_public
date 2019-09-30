#include "pch.h"
#include "SimulationState.h"
#include "helpers.h"
#include "Parameters.h"
#include "Eigen/Dense"
#include <ctime>
#include <random>


std::default_random_engine generator{ static_cast<long unsigned int>(time(nullptr)) };


TEST(TestCreateRotationMatrix, TestBasicRotations) {
	velocities ang_f, ang_b;
	Matrix<double, 1, 3> vec1, vec2;


	// basic rotations around x axis

	ang_f = ang_b = {};
	ang_f.o_x = PI / 2;
	ang_b.o_x = - PI / 2;

	vec1 << 0, 1, 0;
	vec2 << 0, 0, 1;

	EXPECT_TRUE((vec1 * create_rotation_matrix(1, ang_f)).isApprox(vec2));
	EXPECT_TRUE((vec2 * create_rotation_matrix(1, ang_f)).isApprox(-vec1));
	EXPECT_TRUE((vec1 * create_rotation_matrix(1, ang_b)).isApprox(-vec2));
	EXPECT_TRUE((vec2 * create_rotation_matrix(1, ang_b)).isApprox(vec1));


	// basic rotations around y axis

	ang_f = ang_b = {};
	ang_f.o_y = PI / 2;
	ang_b.o_y = -PI / 2;

	vec1 << 1, 0, 0;
	vec2 << 0, 0, 1;

	EXPECT_TRUE((vec1 * create_rotation_matrix(1, ang_f)).isApprox(-vec2));
	EXPECT_TRUE((vec2 * create_rotation_matrix(1, ang_f)).isApprox(vec1));
	EXPECT_TRUE((vec1 * create_rotation_matrix(1, ang_b)).isApprox(vec2));
	EXPECT_TRUE((vec2 * create_rotation_matrix(1, ang_b)).isApprox(-vec1));


	// basic rotations around z axis

	ang_f = ang_b = {};
	ang_f.o_z = PI / 2;
	ang_b.o_z = -PI / 2;

	vec1 << 1, 0, 0;
	vec2 << 0, 1, 0;

	EXPECT_TRUE((vec1 * create_rotation_matrix(1, ang_f)).isApprox(vec2));
	EXPECT_TRUE((vec2 * create_rotation_matrix(1, ang_f)).isApprox(-vec1));
	EXPECT_TRUE((vec1 * create_rotation_matrix(1, ang_b)).isApprox(-vec2));
	EXPECT_TRUE((vec2 * create_rotation_matrix(1, ang_b)).isApprox(vec1));
}

TEST(TestCreateRotationMatrix, TestInversibility) {
	std::uniform_real_distribution<double> uniform_dist(-10, 10);

	velocities random_v = {};
	random_v.o_x = uniform_dist(generator);
	random_v.o_y = uniform_dist(generator);
	random_v.o_z = uniform_dist(generator);
	double random_dt = uniform_dist(generator);

	// any rotation with time -dt should be an inverse of the same rotation with time dt
	Matrix3d wannabe_identity = create_rotation_matrix(random_dt, random_v) * create_rotation_matrix(-random_dt, random_v);
	EXPECT_TRUE(wannabe_identity.isApprox(Matrix3d::Identity()));
}


TEST(TestComputeBondVector, TestRandomBond) {
	Array3d rec_xyz = Array3d::Random();  // [-1, 1]
	Array2d endth_xz = Array2d::Random();

	double& ex = endth_xz(0);
	double& ez = endth_xz(1);
	double& x = rec_xyz(0);
	double& y = rec_xyz(1);
	double& z = rec_xyz(2);
	   
	double r = rec_xyz.matrix().norm();
	std::uniform_real_distribution<double> h_dist(0, r);
	double h = h_dist(generator);

	Vector3d bond_vector;
	bond_vector << -x + ex, -r - h - y, -z + ez;

	EXPECT_TRUE(compute_bond_vector(rec_xyz, endth_xz, h, r).isApprox(bond_vector));
}


class TestLinearInterpolation : public ::testing::Test {
protected:
	void SetUp() override {
		std::uniform_int_distribution<int> table_size_dist(2, 100);
		std::uniform_real_distribution<double> x_inc_dist(0, 3);
		std::uniform_real_distribution<double> val_diff_dist(-10, 10);
		
		table_size = table_size_dist(generator);
		points.resize(table_size, NoChange);

		points(0, 0) = x_inc_dist(generator);
		points(0, 1) = val_diff_dist(generator);
		for (int i = 1; i < table_size; i++) {
			points(i, 0) = points(i - 1, 0) + x_inc_dist(generator);
			points(i, 1) = points(i - 1, 1) + val_diff_dist(generator);
		}
	}

	int table_size;
	ArrayX2d points;
};

TEST_F(TestLinearInterpolation, TestExactTableValues) {
	for (int i = 0; i < table_size; i++)
		// TODO: DOUBLE_EQ requires precision that is not always achieved.
		EXPECT_FLOAT_EQ(linear_interpolation(points, points(i, 0)), points(i, 1));
}

TEST_F(TestLinearInterpolation, TestValuesInTheMiddle) {
	for (int i = 0; i < table_size - 1; i++)
		// TODO: DOUBLE_EQ requires precision that is not always achieved.
		EXPECT_FLOAT_EQ(
			linear_interpolation(points, (points(i, 0) + points(i + 1, 0)) / 2),
			(points(i, 1) + points(i + 1, 1)) / 2
		);
}


class TestCoordsFromCoefs : public ::testing::Test {
protected:
	void SetUp() override {
		std::uniform_int_distribution<int> n_of_rec_dist(1, 20000);
		n_of_rec = n_of_rec_dist(generator);
		lig_coef.setRandom(n_of_rec, 3);  // NoChange doesn't work here
	}

	int n_of_rec;
	Matrix3d ref_vec;
	ArrayX3d lig_coef;

	void with_scaled_standard_basis(double scale) {
		ref_vec = Matrix3d::Identity() * scale;

		for (int i = 0; i < n_of_rec; i++)
			for (int j = 0; j < 3; j++)
				EXPECT_DOUBLE_EQ(
					get_coord_from_coef(ref_vec, lig_coef.row(i), j),
					scale * lig_coef(i, j)
				);

		for (int j = 0; j < 3; j++)
			EXPECT_TRUE(get_coord_from_coefs(ref_vec, lig_coef, j).isApprox(scale * lig_coef.col(j)));

		ArrayX2d coords;
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++) {
				coords = get_coords_from_coefs(ref_vec, lig_coef, j, k);
				EXPECT_TRUE(coords.col(0).isApprox(scale * lig_coef.col(j)));
				EXPECT_TRUE(coords.col(1).isApprox(scale * lig_coef.col(k)));
			}

		EXPECT_TRUE(get_coords_from_coefs(ref_vec, lig_coef).isApprox(scale * lig_coef));
	}
};

TEST_F(TestCoordsFromCoefs, TestStandardBasis) {
	with_scaled_standard_basis(1);
}

TEST_F(TestCoordsFromCoefs, TestScaledStandardBasis) {
	std::uniform_real_distribution<double> scale_dist(0, 10);
	double scale = scale_dist(generator);
	with_scaled_standard_basis(scale);
}

TEST_F(TestCoordsFromCoefs, TestRandomBasis) {
	ref_vec.setRandom();
	auto& v0 = ref_vec.row(0);
	auto& v1 = ref_vec.row(1);
	auto& v2 = ref_vec.row(2);

	for (int i = 0; i < n_of_rec; i++)
		for (int j = 0; j < 3; j++)
			EXPECT_DOUBLE_EQ(
				get_coord_from_coef(ref_vec, lig_coef.row(i), j),
				lig_coef(i, 0) * v0(j) + lig_coef(i, 1) * v1(j) + lig_coef(i, 2) * v2(j)
			);

	ArrayXd coord;

	for (int j = 0; j < 3; j++) {
		coord = get_coord_from_coefs(ref_vec, lig_coef, j);
		for (int i = 0; i < n_of_rec; i++)
			EXPECT_DOUBLE_EQ(
				coord(i),
				lig_coef(i, 0) * v0(j) + lig_coef(i, 1) * v1(j) + lig_coef(i, 2) * v2(j)
			);
		}

	ArrayX2d coords;
	for (int j = 0; j < 3; j++)
		for (int k = 0; k < 3; k++) {
			coords = get_coords_from_coefs(ref_vec, lig_coef, j, k);
			for (int i = 0; i < n_of_rec; i++) {
				EXPECT_DOUBLE_EQ(
					coords(i, 0),
					lig_coef(i, 0) * v0(j) + lig_coef(i, 1) * v1(j) + lig_coef(i, 2) * v2(j)
				);
				EXPECT_DOUBLE_EQ(
					coords(i, 1),
					lig_coef(i, 0) * v0(k) + lig_coef(i, 1) * v1(k) + lig_coef(i, 2) * v2(k)
				);
			}
		}

	auto all_coords = get_coords_from_coefs(ref_vec, lig_coef);
	for (int i = 0; i < n_of_rec; i++)
		for (int j = 0; i < 3; i++) 
			EXPECT_DOUBLE_EQ(
				all_coords(i, j),
				lig_coef(i, 0) * v0(j) + lig_coef(i, 1) * v1(j) + lig_coef(i, 2) * v2(j)
			);
}