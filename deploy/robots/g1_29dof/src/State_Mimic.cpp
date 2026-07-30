#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;

Eigen::Quaternionf torso_quat_w(isaaclab::ManagerBasedRLEnv* env) {
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    auto & motors = robot->lowstate->msg_.motor_state();

    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY()) \
    ;
    return torso_quat;
};

Eigen::Quaternionf anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(joint_pos[13], Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(joint_pos[14], Eigen::Vector3f::UnitY()) \
    ;
    return torso_quat;
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_joint_pos)
{
    auto & robot = env->robot;
    auto & loader = State_Mimic::motion;
    auto & ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_pos();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for(int i = 0; i < data_dfs.size(); ++i) {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_joint_vel)
{
    auto & robot = env->robot;
    auto & loader = State_Mimic::motion;
    auto & ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_vel();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for(int i = 0; i < data_dfs.size(); ++i) {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_command)
{
    auto & robot = env->robot;
    auto & loader = State_Mimic::motion;
    auto & ids = robot->data.joint_ids_map;

    auto pos_dfs = loader->joint_pos();
    Eigen::VectorXf pos_bfs = Eigen::VectorXf::Zero(pos_dfs.size());
    for(int i = 0; i < pos_dfs.size(); ++i) {
        pos_bfs(i) = pos_dfs[ids[i]];
    }
    auto vel_dfs = loader->joint_vel();
    Eigen::VectorXf vel_bfs = Eigen::VectorXf::Zero(vel_dfs.size());
    for(int i = 0; i < vel_dfs.size(); ++i) {
        vel_bfs(i) = vel_dfs[ids[i]];
    }
    std::vector<float> data;
    data.insert(data.end(), pos_bfs.data(), pos_bfs.data() + pos_bfs.size());
    data.insert(data.end(), vel_bfs.data(), vel_bfs.data() + vel_bfs.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    // auto & robot = env->robot;
    auto real_quat_w = torso_quat_w(env);
    auto ref_quat_w = anchor_quat_w(State_Mimic::motion);

    Eigen::Quaternionf rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    // Materialize the matrix: with `auto`, transpose() is a lazy expression that
    // references the temporary returned by toRotationMatrix(), which is destroyed
    // at the end of this statement. Reading rot(i,j) afterwards is use-after-scope
    // and fills the observation with garbage.
    Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if(!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    // Motion
    motion_ = std::make_shared<MotionLoader_>(motion_file.string(), cfg["fps"].as<float>());
    spdlog::info("Loaded motion file '{}' with duration {:.2f}s", motion_file.stem().string(), motion_->duration);
    motion = motion_;
    if(cfg["time_start"]) {
        float time_start = cfg["time_start"].as<float>();
        time_range_[0] = std::clamp(time_start, 0.0f, motion_->duration);
    } else {
        time_range_[0] = 0.0f;
    }
    if(cfg["time_end"]) {
        float time_end = cfg["time_end"].as<float>();
        time_range_[1] = std::clamp(time_end, 0.0f, motion_->duration);
    } else {
        time_range_[1] = motion_->duration;
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    const auto & joy = FSMState::lowstate->joystick;
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (env->episode_length * env->step_dt) > time_range_[1]; }, // time out
            FSMStringMap.right.at("Velocity")
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); }, // bad orientation
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Mimic::enter()
{
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    motion = motion_; // set for specific motion
    // Refresh robot data before any anchor math (State_RLBase::enter() does this
    // too). On the first entry data.root_quat_w has never been filled, so
    // init_quat and motion->reset() would otherwise use uninitialized data.
    env->robot->update();
    env->reset();
    // Hold the measured pose until the policy computes its first action;
    // otherwise run() pushes zero/stale targets for the first control ticks.
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
        int idx = env->robot->data.joint_ids_map[i];
        lowcmd->msg_.motor_cmd()[idx].q() = lowstate->msg_.motor_state()[idx].q();
    }
    first_action_ = false;
    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        auto ref_yaw = isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();
        auto robot_yaw = isaaclab::yawQuaternion(torso_quat_w(env.get())).toRotationMatrix();
        init_quat = robot_yaw * ref_yaw.transpose();
        motion->reset(env->robot->data, time_range_[0]);
        env->reset();

        while (policy_thread_running)
        {
            env->robot->update();
            motion->update(env->episode_length * env->step_dt + time_range_[0]);
            env->step();
            first_action_ = true;

            // Sleep
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Mimic::run()
{
    if (!first_action_) return; // keep holding the entry pose set in enter()
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}