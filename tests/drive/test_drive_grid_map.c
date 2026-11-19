#include "include/drive_fixture.h"
#include "include/test.h"

#define TEST_POINT_COUNT 2
#define TEST_MAP_NAME "grid_map_test"
#define TEST_ROAD_SPAN_METERS 100.0f
// Large enough that the extent in cells exceeds MAX_GRID_CELL_COUNT in a single dimension.
#define TEST_HUGE_SPAN_METERS (GRID_CELL_SIZE * (float) MAX_GRID_CELL_COUNT * 4.0f)

// Builds a Drive holding one road edge running through the two given points.
static Drive make_edge_env(float *x_points, float *y_points) {
    static RoadMapElement element;
    memset(&element, 0, sizeof(element));
    element.type = ROAD_EDGE_BOUNDARY;
    element.segment_size = TEST_POINT_COUNT;
    element.x = x_points;
    element.y = y_points;

    Drive env = {0};
    env.road_elements = &element;
    env.num_road_elements = 1;
    env.map_name = TEST_MAP_NAME;
    return env;
}

static void free_grid_map(Drive *env) {
    GridMap *grid_map = env->grid_map;
    int grid_cell_count = grid_map->grid_cols * grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(grid_map->cells[grid_index]);
    }
    free(grid_map->cells);
    free(grid_map->cell_entities_count);
    free(grid_map->grid_index_drivable);
    free(grid_map);
}

static int test_grid_map_valid_extent(void) {
    float x_points[TEST_POINT_COUNT] = {0.0f, TEST_ROAD_SPAN_METERS};
    float y_points[TEST_POINT_COUNT] = {0.0f, TEST_ROAD_SPAN_METERS};
    Drive env = make_edge_env(x_points, y_points);

    EXPECT_EQ_INT(init_grid_map(&env), 0);
    EXPECT_EQ_INT(env.grid_map->grid_cols, (int) (TEST_ROAD_SPAN_METERS / GRID_CELL_SIZE));
    EXPECT_EQ_INT(env.grid_map->grid_rows, (int) (TEST_ROAD_SPAN_METERS / GRID_CELL_SIZE));

    free_grid_map(&env);
    return 0;
}

static int test_grid_map_rejects_zero_extent(void) {
    float x_points[TEST_POINT_COUNT] = {5.0f, 5.0f};
    float y_points[TEST_POINT_COUNT] = {5.0f, 5.0f};
    Drive env = make_edge_env(x_points, y_points);

    EXPECT_EQ_INT(init_grid_map(&env), -1);
    return 0;
}

static int test_grid_map_rejects_axis_aligned_line(void) {
    float x_points[TEST_POINT_COUNT] = {0.0f, TEST_ROAD_SPAN_METERS};
    float y_points[TEST_POINT_COUNT] = {5.0f, 5.0f};
    Drive env = make_edge_env(x_points, y_points);

    EXPECT_EQ_INT(init_grid_map(&env), -1);
    return 0;
}

static int test_grid_map_rejects_no_road_geometry(void) {
    float x_points[TEST_POINT_COUNT] = {INVALID_POSITION, INVALID_POSITION};
    float y_points[TEST_POINT_COUNT] = {INVALID_POSITION, INVALID_POSITION};
    Drive env = make_edge_env(x_points, y_points);

    EXPECT_EQ_INT(init_grid_map(&env), -1);
    return 0;
}

static int test_grid_map_rejects_oversized_extent(void) {
    float x_points[TEST_POINT_COUNT] = {0.0f, TEST_HUGE_SPAN_METERS};
    float y_points[TEST_POINT_COUNT] = {0.0f, TEST_ROAD_SPAN_METERS};
    Drive env = make_edge_env(x_points, y_points);

    EXPECT_EQ_INT(init_grid_map(&env), -1);
    return 0;
}

static int test_grid_map_rejects_oversized_cell_product(void) {
    // Each dimension is under MAX_GRID_CELL_COUNT but their product is not.
    float span_meters = GRID_CELL_SIZE * (float) (MAX_GRID_CELL_COUNT / 2);
    float x_points[TEST_POINT_COUNT] = {0.0f, span_meters};
    float y_points[TEST_POINT_COUNT] = {0.0f, span_meters};
    Drive env = make_edge_env(x_points, y_points);

    EXPECT_EQ_INT(init_grid_map(&env), -1);
    return 0;
}

static int test_grid_map_rejects_non_finite_extent(void) {
    float x_points[TEST_POINT_COUNT] = {0.0f, INFINITY};
    float y_points[TEST_POINT_COUNT] = {0.0f, TEST_ROAD_SPAN_METERS};
    Drive env = make_edge_env(x_points, y_points);
    EXPECT_EQ_INT(init_grid_map(&env), -1);

    x_points[1] = NAN;
    EXPECT_EQ_INT(init_grid_map(&env), -1);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_grid_map_valid_extent);
    RUN_TEST(test_grid_map_rejects_zero_extent);
    RUN_TEST(test_grid_map_rejects_axis_aligned_line);
    RUN_TEST(test_grid_map_rejects_no_road_geometry);
    RUN_TEST(test_grid_map_rejects_oversized_extent);
    RUN_TEST(test_grid_map_rejects_oversized_cell_product);
    RUN_TEST(test_grid_map_rejects_non_finite_extent);
    return test_summary(failures);
}
