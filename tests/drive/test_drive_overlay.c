#include "include/drive_fixture.h"
#include "include/test.h"

// One agent at (10, 5) heading +90 degrees, the followed one
static Drive overlay_test_env(Agent *agent, int *active_agent_index) {
    Drive env = {0};
    *agent = drive_test_agent(10.0f, 5.0f, 0.5f * (float) M_PI);
    *active_agent_index = 0;
    env.agents = agent;
    env.active_agent_indices = active_agent_index;
    env.active_agent_count = 1;
    env.human_agent_idx = 0;
    return env;
}

static float *overlay_node(float *row, int node_idx) {
    return &row[OVERLAY_HEADER + node_idx * OVERLAY_NODE_FLOATS];
}

static int test_overlay_converts_ego_frame_to_world(void) {
    Agent agent;
    int active_agent_index;
    Drive env = overlay_test_env(&agent, &active_agent_index);
    float row[OVERLAY_FLOATS] = {0};
    row[OVERLAY_VALID_IDX] = 1.0f;
    row[OVERLAY_NODE_COUNT_IDX] = 2.0f;
    row[OVERLAY_PLANE_COUNT_IDX] = 1.0f;
    float *node = overlay_node(row, 1);
    node[OVERLAY_NODE_X_IDX] = 1.0f;
    node[OVERLAY_NODE_Y_IDX] = 0.0f;
    node[OVERLAY_NODE_TARGET_Y_IDX] = 0.5f;
    node[OVERLAY_NODE_WEIGHT_Y_IDX] = 2.0f;
    // Left plane p_y <= 2 in the ego frame
    node[OVERLAY_NODE_PLANES_IDX] = 0.0f;
    node[OVERLAY_NODE_PLANES_IDX + 1] = 1.0f;
    node[OVERLAY_NODE_PLANES_IDX + 2] = 2.0f;
    EXPECT_EQ_INT(c_set_overlay(&env, row), 0);
    Overlay *overlay = &env.overlay;
    EXPECT_EQ_INT(overlay->valid, 1);
    EXPECT_EQ_INT(overlay->node_count, 2);
    EXPECT_EQ_INT(overlay->plane_count, 1);
    // Ego x points along world +y, ego y along world -x
    EXPECT_NEAR(overlay->position[1][0], 10.0f, 1e-5f);
    EXPECT_NEAR(overlay->position[1][1], 6.0f, 1e-5f);
    EXPECT_NEAR(overlay->cost_center[1][0], 9.5f, 1e-5f);
    EXPECT_NEAR(overlay->cost_center[1][1], 6.0f, 1e-5f);
    EXPECT_NEAR(overlay->weight_y[1], 2.0f, 1e-6f);
    EXPECT_NEAR(overlay->lateral_x, -1.0f, 1e-5f);
    EXPECT_NEAR(overlay->lateral_y, 0.0f, 1e-5f);
    // World plane -x <= b_w: the ego-frame point (0, 2) is (8, 5)
    float *plane = overlay->plane[1][0];
    EXPECT_NEAR(plane[0], -1.0f, 1e-5f);
    EXPECT_NEAR(plane[1], 0.0f, 1e-5f);
    EXPECT_NEAR(plane[0] * 8.0f + plane[1] * 5.0f, plane[2], 1e-4f);
    return 0;
}

static int test_overlay_rejects_bad_rows(void) {
    Agent agent;
    int active_agent_index;
    Drive env = overlay_test_env(&agent, &active_agent_index);
    float row[OVERLAY_FLOATS] = {0};
    row[OVERLAY_VALID_IDX] = 1.0f;
    row[OVERLAY_NODE_COUNT_IDX] = (float) (OVERLAY_MAX_NODES + 1);
    EXPECT_EQ_INT(c_set_overlay(&env, row), -1);
    row[OVERLAY_NODE_COUNT_IDX] = 2.0f;
    row[OVERLAY_PLANE_COUNT_IDX] = 1.5f;
    EXPECT_EQ_INT(c_set_overlay(&env, row), -1);
    row[OVERLAY_PLANE_COUNT_IDX] = 0.0f;
    row[OVERLAY_VALID_IDX] = 2.0f;
    EXPECT_EQ_INT(c_set_overlay(&env, row), -1);
    row[OVERLAY_VALID_IDX] = 1.0f;
    overlay_node(row, 0)[OVERLAY_NODE_X_IDX] = NAN;
    EXPECT_EQ_INT(c_set_overlay(&env, row), -1);
    return 0;
}

static int test_invalid_overlay_draws_nothing(void) {
    Agent agent;
    int active_agent_index;
    Drive env = overlay_test_env(&agent, &active_agent_index);
    float row[OVERLAY_FLOATS] = {0};
    EXPECT_EQ_INT(c_set_overlay(&env, row), 0);
    EXPECT_EQ_INT(env.overlay.valid, 0);
    return 0;
}

// Two active agents: the followed one at (10, 5) heading +90 degrees, the
// other at (0, 0) heading 0; each row is in its own agent's ego frame
static int test_overlay_paths_convert_every_agent(void) {
    Agent agents[2];
    agents[0] = drive_test_agent(10.0f, 5.0f, 0.5f * (float) M_PI);
    agents[1] = drive_test_agent(0.0f, 0.0f, 0.0f);
    int active_agent_indices[2] = {0, 1};
    Drive env = {0};
    env.agents = agents;
    env.active_agent_indices = active_agent_indices;
    env.active_agent_count = 2;
    env.human_agent_idx = 0;
    float rows[2 * OVERLAY_FLOATS] = {0};
    for (int i = 0; i < 2; i++) {
        float *row = &rows[i * OVERLAY_FLOATS];
        row[OVERLAY_VALID_IDX] = 1.0f;
        row[OVERLAY_NODE_COUNT_IDX] = 2.0f;
        overlay_node(row, 1)[OVERLAY_NODE_X_IDX] = 1.0f;
        overlay_node(row, 1)[OVERLAY_NODE_Y_IDX] = 0.5f;
    }
    EXPECT_EQ_INT(c_set_overlay_paths(&env, rows), 0);
    EXPECT_EQ_INT(env.overlay_path_count, 2);
    EXPECT_EQ_INT(env.overlay_paths[1].node_count, 2);
    // Ego x along world +y for agent 0, along world +x for agent 1
    EXPECT_NEAR(env.overlay_paths[0].position[1][0], 9.5f, 1e-5f);
    EXPECT_NEAR(env.overlay_paths[0].position[1][1], 6.0f, 1e-5f);
    EXPECT_NEAR(env.overlay_paths[1].position[1][0], 1.0f, 1e-5f);
    EXPECT_NEAR(env.overlay_paths[1].position[1][1], 0.5f, 1e-5f);
    // An invalid row has no path; a bad row drops all paths
    rows[OVERLAY_FLOATS + OVERLAY_VALID_IDX] = 0.0f;
    EXPECT_EQ_INT(c_set_overlay_paths(&env, rows), 0);
    EXPECT_EQ_INT(env.overlay_paths[1].node_count, 0);
    rows[OVERLAY_FLOATS + OVERLAY_NODE_COUNT_IDX] = 1.5f;
    EXPECT_EQ_INT(c_set_overlay_paths(&env, rows), -1);
    EXPECT_EQ_INT(env.overlay_path_count, 0);
    free(env.overlay_paths);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_overlay_converts_ego_frame_to_world);
    RUN_TEST(test_overlay_rejects_bad_rows);
    RUN_TEST(test_invalid_overlay_draws_nothing);
    RUN_TEST(test_overlay_paths_convert_every_agent);
    return test_summary(failures);
}
