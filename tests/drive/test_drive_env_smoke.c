#include <stdint.h>

#include "include/drive_fixture.h"
#include "include/test.h"

static int run_case(const char *name, const char *map_file, int simulation_mode, int num_agents) {
    srand(7);
    Drive env = drive_test_make_env(map_file, simulation_mode, num_agents, 0);
    int obs_size = compute_observation_size(&env);

    EXPECT_TRUE(env.active_agent_count > 0);
    EXPECT_TRUE(env.observations != NULL);
    EXPECT_TRUE(env.rewards != NULL);
    EXPECT_TRUE(drive_all_finite(env.observations, env.active_agent_count * obs_size));

    int saw_log = 0;
    for (int t = 0; t < env.scenario_length + 5; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_TRUE(drive_all_finite(env.observations, env.active_agent_count * obs_size));
        EXPECT_TRUE(drive_all_finite(env.rewards, env.active_agent_count));
        if (env.completed_episodes_count > 0 || env.log.n > 0.0f) {
            saw_log = 1;
        }
        for (int i = 0; i < env.active_agent_count; i++) {
            Agent *agent = &env.agents[env.active_agent_indices[i]];
            int terminal_flags = (agent->metrics_array[COLLISION_IDX] > 0.0f)
                + (agent->metrics_array[OFFROAD_IDX] > 0.0f) + (agent->metrics_array[RED_LIGHT_IDX] > 0.0f);
            EXPECT_TRUE(terminal_flags <= 1);
        }
    }

    EXPECT_TRUE(saw_log);
    printf(
        "case %s active=%d log_n=%.0f completed=%d\n",
        name,
        env.active_agent_count,
        env.log.n,
        env.completed_episodes_count);
    free_allocated(&env);
    return 0;
}

static int test_carla_gigaflow_load_step_log(void) {
    return run_case("carla-gigaflow", drive_carla_map(), SIMULATION_GIGAFLOW, 32);
}

static int test_nuplan_gigaflow_load_step_log(void) {
    return run_case("nuplan-gigaflow", drive_nuplan_map(), SIMULATION_GIGAFLOW, 32);
}

static int test_nuplan_replay_load_step_log(void) {
    return run_case("nuplan-replay", drive_nuplan_map(), SIMULATION_REPLAY, 1);
}

// Replay waypoints stay on the valid part of each agent's log: the last one
// is the last valid logged position, even for agents that leave the log early.
static int test_nuplan_replay_goals_on_valid_log(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_REPLAY, 0, 0);
    env.control_mode = CONTROL_VEHICLES;
    env.scenario_length = 200;
    allocate(&env);
    c_reset(&env);

    EXPECT_TRUE(env.active_agent_count > 1);
    int early_leaver_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        int last_valid_step = agent->trajectory_size - 1;
        for (int t = agent->trajectory_size - 1; t > 0 && agent->log_valid[t] != 1; t--) {
            last_valid_step = t - 1;
        }
        early_leaver_count += last_valid_step < agent->trajectory_size - 1;
        int last_goal_idx = env.num_goals - 1;
        EXPECT_NEAR(agent->list_goal_x[last_goal_idx], agent->log_trajectory_x[last_valid_step], 1e-5f);
        EXPECT_NEAR(agent->list_goal_y[last_goal_idx], agent->log_trajectory_y[last_valid_step], 1e-5f);
        for (int g = 0; g < env.num_goals; g++) {
            int goal_at_origin = agent->list_goal_x[g] == 0.0f && agent->list_goal_y[g] == 0.0f;
            EXPECT_TRUE(!goal_at_origin);
        }
    }
    // The checked-in bin has a controlled vehicle that leaves the log early
    EXPECT_TRUE(early_leaver_count > 0);

    free_allocated(&env);
    return 0;
}

// The checked-in nuPlan bin's logs have 201 frames (indices 0 .. 200)
static int test_replay_episode_fits_log(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_REPLAY, 0, 0);
    EXPECT_EQ_INT(load_map_binary(env.map_name, &env), 0);
    EXPECT_EQ_INT(scene_log_frame_count(&env), 201);
    env.init_step = 0;
    env.scenario_length = 200;
    EXPECT_TRUE(replay_episode_fits_log(&env));
    env.scenario_length = 201;
    EXPECT_FALSE(replay_episode_fits_log(&env));
    env.init_step = 1;
    env.scenario_length = 200;
    EXPECT_FALSE(replay_episode_fits_log(&env));
    env.init_step = 100;
    env.scenario_length = 100;
    EXPECT_TRUE(replay_episode_fits_log(&env));
    for (int i = 0; i < env.num_total_agents; i++) {
        free_agent(&env.agents[i]);
    }
    for (int i = 0; i < env.num_road_elements; i++) {
        free_road_element(&env.road_elements[i]);
    }
    for (int i = 0; i < env.num_traffic_elements; i++) {
        free_traffic_element(&env.traffic_elements[i]);
    }
    free(env.agents);
    free(env.road_elements);
    free(env.traffic_elements);
    free(env.map_name);
    return 0;
}

// Replay self-play on the checked-in nuPlan bin, every unflagged vehicle under the policy
static Drive make_nuplan_replay_env(int replay_replan_goals) {
    srand(17);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_REPLAY, 0, 0);
    env.control_mode = CONTROL_VEHICLES;
    env.scenario_length = 200;
    env.replay_replan_goals = replay_replan_goals;
    allocate(&env);
    c_reset(&env);
    return env;
}

// Puts agent's current goal at (forward_m, 0) in its own frame
static void place_goal_in_ego_frame(Agent *agent, float forward_m) {
    agent->current_goal_x = agent->sim_x + forward_m * cosf(agent->sim_heading);
    agent->current_goal_y = agent->sim_y + forward_m * sinf(agent->sim_heading);
    agent->current_goal_z = agent->sim_z;
    agent->list_goal_x[agent->current_goal_idx] = agent->current_goal_x;
    agent->list_goal_y[agent->current_goal_idx] = agent->current_goal_y;
    agent->list_goal_z[agent->current_goal_idx] = agent->current_goal_z;
}

// Reaching the last log goal gives route goals ahead of the agent; where the
// lane graph ends too early the goal stays saturated and the agent stays
static int test_replay_replan_after_last_log_goal(void) {
    Drive env = make_nuplan_replay_env(1);
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        agent->current_goal_idx = env.num_goals - 1;
        place_goal_in_ego_frame(agent, 0.0f);
    }
    drive_set_neutral_actions(&env);
    c_step(&env);

    int replanned_count = 0;
    int failed_count = 0;
    int skipped_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        EXPECT_FALSE(agent->removed);
        // Agents that collided or went off-road at the first step skip the goal check
        if (agent->metrics_array[REACHED_GOAL_IDX] == 0.0f) {
            EXPECT_EQ_INT(agent->current_goal_idx, env.num_goals - 1);
            skipped_count++;
            continue;
        }
        replanned_count += env.logs[i].replay_replans > 0.0f;
        failed_count += env.logs[i].replay_replan_failures > 0.0f;
        if (env.logs[i].replay_replan_failures > 0.0f) {
            EXPECT_EQ_INT(agent->current_goal_idx, env.num_goals);
            EXPECT_TRUE(agent->replay_replan_failed);
            continue;
        }
        EXPECT_NEAR(env.logs[i].replay_replans, 1.0f, 1e-6f);
        EXPECT_TRUE(agent->replay_goals_replanned);
        EXPECT_EQ_INT(agent->current_goal_idx, 0);
        // Goals are spaced along the route, at least min_goal_spacing apart
        float previous_x = agent->sim_x;
        float previous_y = agent->sim_y;
        for (int g = 0; g < env.num_goals; g++) {
            float dx = agent->list_goal_x[g] - previous_x;
            float dy = agent->list_goal_y[g] - previous_y;
            EXPECT_TRUE(sqrtf(dx * dx + dy * dy) > 0.5f * env.min_goal_spacing);
            previous_x = agent->list_goal_x[g];
            previous_y = agent->list_goal_y[g];
        }
    }
    EXPECT_EQ_INT(replanned_count + failed_count + skipped_count, env.active_agent_count);
    EXPECT_TRUE(replanned_count > 0);
    EXPECT_TRUE(skipped_count < env.active_agent_count / 2);
    printf(
        "replan: %d of %d agents replanned, %d failed, %d skipped\n",
        replanned_count,
        env.active_agent_count,
        failed_count,
        skipped_count);

    // A failed agent does not retry; a reset brings back the log goals
    for (int t = 0; t < 5; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_TRUE(env.logs[i].replay_replan_failures <= 1.0f);
    }
    c_reset(&env);
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        EXPECT_FALSE(agent->replay_goals_replanned);
        EXPECT_FALSE(agent->replay_replan_failed);
        EXPECT_TRUE(agent->path == NULL);
    }
    free_allocated(&env);
    return 0;
}

// A log goal behind the agent is skipped without a goal reward when
// replanning is on, and kept when it is off
static int test_replay_passed_log_goal(void) {
    for (int replay_replan_goals = 0; replay_replan_goals <= 1; replay_replan_goals++) {
        Drive env = make_nuplan_replay_env(replay_replan_goals);
        // A moving vehicle still before its first log goal (stopped ones reach it at reset)
        int log_idx = -1;
        for (int i = 0; i < env.active_agent_count && log_idx == -1; i++) {
            Agent *candidate = &env.agents[env.active_agent_indices[i]];
            if (candidate->current_goal_idx == 0 && candidate->sim_speed > 1.0f) {
                log_idx = i;
            }
        }
        EXPECT_TRUE(log_idx != -1);
        Agent *agent = &env.agents[env.active_agent_indices[log_idx]];
        place_goal_in_ego_frame(agent, -5.0f);
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_EQ_INT(agent->current_goal_idx, replay_replan_goals);
        EXPECT_NEAR(env.logs[log_idx].replay_goals_passed, (float) replay_replan_goals, 1e-6f);
        EXPECT_NEAR(env.logs[log_idx].num_waypoints_reached, 0.0f, 1e-6f);
        free_allocated(&env);
    }
    return 0;
}

// With the facing filter, the first traffic-control slot holds the nearest
// in-range light whose heading is within STOP_LINE_HEADING_THRESHOLD; with
// unknown-as-off no light slot reads the unknown state
static int test_traffic_control_obs_facing_and_unknown(void) {
    Drive env = make_nuplan_replay_env(0);
    env.traffic_control_obs_facing_only = 1;
    env.traffic_light_obs_unknown_as_off = 1;
    float obs[TRAFFIC_CONTROL_FEATURES * 4];
    int checked_count = 0;
    for (int t = 0; t < 100; t++) {
        for (int i = 0; i < env.active_agent_count; i++) {
            Agent *ego = &env.agents[env.active_agent_indices[i]];
            int expected_idx = -1;
            float expected_dist_sq = env.obs_range_traffic_control_m * env.obs_range_traffic_control_m;
            for (int j = 0; j < env.num_traffic_elements; j++) {
                TrafficControlElement *tc = &env.traffic_elements[j];
                if (!traffic_control_in_scope(tc->type, env.traffic_control_scope)
                    || fabsf(compute_heading_diff(ego->sim_heading, tc->heading)) > STOP_LINE_HEADING_THRESHOLD) {
                    continue;
                }
                float dx = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f - ego->sim_x;
                float dy = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f - ego->sim_y;
                float dz = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f - ego->sim_z;
                float dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq <= expected_dist_sq && fabsf(dz) <= Z_BUFFER) {
                    expected_dist_sq = dist_sq;
                    expected_idx = j;
                }
            }
            int control_count = 0;
            write_traffic_control_obs(&env, ego, obs, 0, &control_count);
            for (int k = 0; k < control_count; k++) {
                EXPECT_TRUE(obs[k * TRAFFIC_CONTROL_FEATURES + 6] != TRAFFIC_CONTROL_STATE_UNKNOWN);
            }
            if (expected_idx == -1) {
                EXPECT_EQ_INT(control_count, 0);
                continue;
            }
            TrafficControlElement *expected = &env.traffic_elements[expected_idx];
            float rel_x, rel_y;
            project_point_to_ego_frame(ego, expected->stop_line[0], expected->stop_line[1], &rel_x, &rel_y);
            EXPECT_NEAR(obs[0], rel_x / env.obs_norm_xy_offset_m, 1e-5f);
            EXPECT_NEAR(obs[1], rel_y / env.obs_norm_xy_offset_m, 1e-5f);
            checked_count++;
        }
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    EXPECT_TRUE(checked_count > 0);
    free_allocated(&env);
    return 0;
}

#define CACHE_CHECK_STEP_COUNT 200

// FNV-1a over the bytes of every active agent's observation
static uint64_t hash_observations(const Drive *env) {
    const unsigned char *bytes = (const unsigned char *) env->observations;
    size_t byte_count = (size_t) env->active_agent_count * compute_observation_size((Drive *) env) * sizeof(float);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < byte_count; i++) {
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    return hash;
}

static void run_observation_hashes(
    const char *map_file,
    int simulation_mode,
    int num_agents,
    int use_neighbor_cache,
    uint64_t *hashes) {
    srand(13);
    Drive env = drive_test_env_config(map_file, simulation_mode, num_agents, 0);
    env.use_neighbor_cache = use_neighbor_cache;
    if (simulation_mode == SIMULATION_REPLAY) {
        env.control_mode = CONTROL_VEHICLES;
    }
    env.scenario_length = CACHE_CHECK_STEP_COUNT;
    allocate(&env);
    c_reset(&env);
    hashes[0] = hash_observations(&env);
    for (int t = 1; t <= CACHE_CHECK_STEP_COUNT; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        hashes[t] = hash_observations(&env);
    }
    free_allocated(&env);
}

// Gathering road entities per query must give the cached observations bit for bit
static int check_neighbor_cache_off_matches(const char *map_file, int simulation_mode, int num_agents) {
    uint64_t cached[CACHE_CHECK_STEP_COUNT + 1];
    uint64_t gathered[CACHE_CHECK_STEP_COUNT + 1];
    run_observation_hashes(map_file, simulation_mode, num_agents, 1, cached);
    run_observation_hashes(map_file, simulation_mode, num_agents, 0, gathered);
    for (int t = 0; t <= CACHE_CHECK_STEP_COUNT; t++) {
        EXPECT_TRUE(cached[t] == gathered[t]);
    }
    return 0;
}

static int test_neighbor_cache_off_matches_nuplan_replay(void) {
    return check_neighbor_cache_off_matches(drive_nuplan_map(), SIMULATION_REPLAY, 0);
}

static int test_neighbor_cache_off_matches_carla_gigaflow(void) {
    return check_neighbor_cache_off_matches(drive_carla_map(), SIMULATION_GIGAFLOW, 32);
}

static int test_truncation_and_completed_episode_queue(void) {
    srand(11);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_GIGAFLOW, 8, 0);
    env.scenario_length = 3;
    allocate(&env);
    c_reset(&env);

    for (int t = 0; t < 3; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }

    EXPECT_EQ_INT(env.completed_episodes_count, 1);
    EXPECT_EQ_INT(env.next_episode_index, 1);
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_EQ_INT(env.truncations[i], 1);
    }

    CompletedEpisodeSummary summary = {0};
    EXPECT_EQ_INT(pop_completed_episode_summary(&env, &summary), 1);
    EXPECT_EQ_INT(env.completed_episodes_count, 0);
    EXPECT_EQ_INT(summary.episode_index, 0);
    EXPECT_NEAR(summary.n, (float) env.active_agent_count, 1e-5f);

    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_carla_gigaflow_load_step_log);
    RUN_TEST(test_nuplan_gigaflow_load_step_log);
    RUN_TEST(test_nuplan_replay_load_step_log);
    RUN_TEST(test_nuplan_replay_goals_on_valid_log);
    RUN_TEST(test_truncation_and_completed_episode_queue);
    RUN_TEST(test_replay_episode_fits_log);
    RUN_TEST(test_replay_replan_after_last_log_goal);
    RUN_TEST(test_replay_passed_log_goal);
    RUN_TEST(test_traffic_control_obs_facing_and_unknown);
    RUN_TEST(test_neighbor_cache_off_matches_nuplan_replay);
    RUN_TEST(test_neighbor_cache_off_matches_carla_gigaflow);
    return test_summary(failures);
}
