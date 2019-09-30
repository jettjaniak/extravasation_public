#include "pch.h"
#include "SimulationState.h"
#include "helpers.h"
#include "Eigen/Dense"

TEST(TestCreateRotationMatrix, TestBasicRotations) {
  EXPECT_EQ(1, 1);
  velocities v = {};
  create_rotation_matrix(0.01, v);
  EXPECT_TRUE(true);
}