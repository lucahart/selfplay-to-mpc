#!/bin/bash
# Settings of the paper's runs, sourced by the other scripts in this
# folder. Sizes: small (4x512 teacher) and large (4x1024 teacher).
# Students: mlp, unconstrained (DiffMPC with a control box only) and
# constrained (DiffMPC with road planes and agent ellipses).

PAPER_SIZES="small large"
PAPER_STUDENTS="mlp unconstrained constrained"
PAPER_SEEDS="1 2 3 4 5 6 7 8 9"
# Test sets: evaluator section and scenario count
PAPER_TEST_SETS="validation_gigaflow:1000 nuplan_multi_replan:3645 womd_multi:1000"
# Where runs and evaluations go
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/paper}

# DAgger loop shared by every student: 1024 agents, teacher-driven for the
# first 4M labels, then beta drops by 0.2 every 2M labels; a 250-scenario
# CARLA evaluation every 10M labels, at most 100M labels
LOOP_SETTINGS="learning_rate=3e-4 batch_size=1024 env_steps_per_iteration=32 \
beta_schedule=dagger dagger_warmup_labels=4000000 \
dagger_stage_labels=2000000 dagger_stage_count=5 agent_count=1024 \
iterations=100000 max_labels=100000000 eval_interval_labels=10000000 \
eval_num_scenarios=250"

# Teacher checkpoint and network of a size
teacher_settings() {
  case $1 in
    large) echo "teacher_checkpoint=checkpoints/teacher_4x1024.pt" ;;
    small) echo "teacher_checkpoint=checkpoints/teacher_4x512.pt \
policy.ego_input_size=64 policy.partner_input_size=64 \
policy.lane_input_size=64 policy.boundary_input_size=64 \
policy.traffic_control_input_size=64 policy.context_input_size=64 \
policy.backbone_hidden_size=512 policy.backbone_num_layers=4 \
policy.backbone_layer_norm=false policy.shared_network=true" ;;
    *) echo "unknown size $1" >&2; return 1 ;;
  esac
}

# Student type and head of a student (also needed to load its checkpoint)
student_settings() {
  local solver="student=mpc mpc.init_from_teacher=true \
mpc.horizon_steps=10 mpc.solver=warpmpc mpc.sqp_iterations=5 \
mpc.admm_iterations=25"
  case $1 in
    mlp) echo "student=mlp mlp.init_from_teacher=true" ;;
    unconstrained) echo "$solver" ;;
    constrained) echo "$solver mpc.constraint_type=tube \
mpc.tube_margin_m=0.2 mpc.constraint_count=4 \
mpc.tube_anchor=route mpc.tube_edge_side_filter=true \
mpc.tube_plane_max_offset_m=50 mpc.weight_mode=sum mpc.weight_sum=40 \
mpc.slack_price_mode=sum mpc.node_weight_mode=learned" ;;
    *) echo "unknown student $1" >&2; return 1 ;;
  esac
}

# How a student trains: the MLP from a replay buffer, the DiffMPC students
# on the gradient of the solve that drives each agent
training_settings() {
  case $1 in
    mlp) echo "fresh_gradient=false gradient_steps_per_iteration=32" ;;
    unconstrained|constrained) echo "fresh_gradient=true chunks_per_update=1" ;;
    *) echo "unknown student $1" >&2; return 1 ;;
  esac
}

run_dir() {
  echo "$OUTPUT_ROOT/$1_$2_s$3"
}
