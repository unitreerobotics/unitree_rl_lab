// ###################### OLDER VERSION WITHOUT WEB PORTAL SUPPORT##############################

// #include "FSM/CtrlFSM.h"
// #include "FSM/State_Passive.h"
// #include "FSM/State_FixStand.h"
// #include "FSM/State_RLBase.h"
// #include "FSM/State_Crouch.h"

// std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
// std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
// std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

// void init_fsm_state()
// {
//     auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
//     usleep(0.2 * 1e6);
//     if(!lowcmd_sub->isTimeout())
//     {
//         spdlog::critical("The other process is using the lowcmd channel, please close it first.");
//         unitree::robot::go2::shutdown();
//         // exit(0);
//     }
//     FSMState::lowcmd = std::make_unique<LowCmd_t>();
//     FSMState::lowstate = std::make_shared<LowState_t>();
//     spdlog::info("Waiting for connection to robot...");
//     FSMState::lowstate->wait_for_connection();
//     spdlog::info("Connected to robot.");
// }

// int main(int argc, char** argv)
// {
//     // Load parameters
//     auto vm = param::helper(argc, argv);

//     std::cout << " --- Unitree Robotics --- \n";
//     std::cout << "     Go2 Controller \n";

//     // Unitree DDS Config
//     unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

//     init_fsm_state();

//     // Initialize FSM
//     auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
//     fsm->start();

//     std::cout << "Press [L2 + A] to enter FixStand mode.\n";
//     std::cout << "And then press [Start] to start controlling the robot.\n";

//     while (true)
//     {
//         sleep(1);
//     }
    
//     return 0;
// }
















#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "FSM/State_Crouch.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state(); // defined below main(); forward-declared for use in cmd_run_policy()

namespace
{
    // State ids, from FSM.yaml's "_" block (Passive:1, FixStand:2, Velocity:3, Crouch:4).
    // Update these if you renumber states in your yaml.
    constexpr int kPassiveStateId  = 1;
    constexpr int kFixStandStateId = 2;
    constexpr int kVelocityStateId = 3;
    constexpr int kCrouchStateId   = 4;

    std::atomic<bool> g_dds_connected{false};
    std::atomic<bool> g_fsm_running{false};
    std::atomic<bool> g_exit_in_progress{false};
    std::atomic<bool> g_abort_sequence{false};

    std::mutex g_fsm_mutex;
    std::unique_ptr<CtrlFSM> g_fsm;
    std::string g_network_iface;

    // One line of JSON-ish status out to stdout -> websocketd -> browser.
    void send(const std::string& msg)
    {
        std::cout << msg << std::endl; // std::endl flushes; required so websocketd forwards it promptly
    }

    // Reads the interpolation duration (last value of a state's "ts" array)
    // straight from the loaded FSM.yaml, so the exit sequence waits exactly
    // as long as that state's own pose transition takes. Falls back to
    // fallback_seconds if the entry isn't found/parseable.
    double transition_duration_seconds(const std::string& state_name, double fallback_seconds)
    {
        try
        {
            auto ts_node = param::config["FSM"][state_name]["ts"];
            if (ts_node && ts_node.IsSequence() && ts_node.size() > 0)
            {
                return ts_node[ts_node.size() - 1].as<double>();
            }
        }
        catch (...)
        {
            // fall through to default below
        }
        return fallback_seconds;
    }

    // Sleeps up to `seconds`, waking early if an e-stop sets g_abort_sequence.
    // Returns true if it was interrupted.
    bool interruptible_sleep(double seconds)
    {
        using namespace std::chrono;
        const auto step = milliseconds(100);
        auto remaining = duration_cast<milliseconds>(duration<double>(seconds));
        while (remaining.count() > 0)
        {
            if (g_abort_sequence.load()) return true;
            auto this_step = std::min(step, remaining);
            std::this_thread::sleep_for(this_step);
            remaining -= this_step;
        }
        return g_abort_sequence.load();
    }

    void cmd_connect()
    {
        if (g_dds_connected.load())
        {
            send(R"({"status":"already_connected"})");
            return;
        }

        unitree::robot::ChannelFactory::Instance()->Init(0, g_network_iface);
        g_dds_connected.store(true);
        send(R"({"status":"dds_connected"})");
    }

    void cmd_run_policy()
    {
        std::lock_guard<std::mutex> lock(g_fsm_mutex);

        if (!g_dds_connected.load())
        {
            send(R"({"status":"error","detail":"dds_not_connected"})");
            return;
        }
        if (g_fsm_running.load())
        {
            send(R"({"status":"already_running"})");
            return;
        }

        init_fsm_state();

        g_fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
        g_fsm->start();
        g_fsm_running.store(true);

        send(R"({"status":"fsm_started"})");
        send(R"({"status":"info","detail":"Press [L2 + A] on the controller to enter FixStand, then [Start] to run the velocity policy."})");
    }

    // Walks the FSM back down through FixStand -> Crouch -> Passive (skipping
    // whichever of those the FSM is already past), waiting for each pose's
    // own configured "ts" interpolation time before issuing the next
    // transition. Runs on its own thread so cmd_estop() can interrupt it
    // instantly instead of waiting for it to finish.
    void run_graceful_exit()
    {
        struct Step { int id; const char* name; };
        std::vector<Step> sequence;

        const int start_id = g_fsm->getCurrentStateId();
        switch (start_id)
        {
            case kVelocityStateId:
                sequence = {{kFixStandStateId, "FixStand"}, {kCrouchStateId, "Crouch"}};
                break;
            case kFixStandStateId:
                sequence = {{kCrouchStateId, "Crouch"}};
                break;
            case kCrouchStateId:
                sequence = {}; // already there, nothing to do before unload
                break;
            default:
                sequence = {}; // Passive (or an unrecognized state) - nothing to do before unload
                break;
        }

        for (const auto& step : sequence)
        {
            if (g_abort_sequence.load())
            {
                send(R"({"status":"exit_aborted"})");
                g_exit_in_progress.store(false);
                return;
            }

            g_fsm->requestTransition(step.id);
            send(std::string(R"({"status":"exit_step","detail":")") + step.name + "\"}");

            double wait_s = 0.5; // Passive is the final step; just let it settle briefly
            if (std::string(step.name) == "FixStand")
            {
                wait_s = transition_duration_seconds("FixStand", 2.0) + 0.5;
            }
            else if (std::string(step.name) == "Crouch")
            {
                wait_s = transition_duration_seconds("Crouch", 3.0) + 0.5;
            }

            if (interruptible_sleep(wait_s))
            {
                send(R"({"status":"exit_aborted"})");
                g_exit_in_progress.store(false);
                return;
            }
        }

        send(R"({"status":"exit_complete"})");

        // Fully unload: destroy the CtrlFSM (stops its RecurrentThread and
        // releases the loaded policy) and clear the running flag, so
        // "Run Walking Policy" starts fresh next time -- this is what
        // actually returns things to the pre-run state, not just Passive.
        {
            std::lock_guard<std::mutex> lock(g_fsm_mutex);
            g_fsm.reset();
            g_fsm_running.store(false);
        }
        send(R"({"status":"fsm_unloaded"})");

        g_exit_in_progress.store(false);
    }

    // "Exit walking policy" -> graceful, staged descent (FixStand -> Crouch ->
    // Passive as applicable) rather than an instant drop to Passive.
    void cmd_exit_policy()
    {
        std::lock_guard<std::mutex> lock(g_fsm_mutex);

        if (!g_fsm_running.load() || !g_fsm)
        {
            send(R"({"status":"error","detail":"fsm_not_running"})");
            return;
        }
        if (g_exit_in_progress.load())
        {
            send(R"({"status":"error","detail":"exit_already_in_progress"})");
            return;
        }

        g_abort_sequence.store(false);
        g_exit_in_progress.store(true);
        send(R"({"status":"exit_requested"})");

        std::thread(run_graceful_exit).detach();
    }

    // Emergency stop -> forces Passive immediately, on this tick, no staging.
    // Also tells any in-flight graceful-exit sequence to stop issuing further
    // transitions so the two can't fight each other.
    void cmd_estop()
    {
        g_abort_sequence.store(true);

        std::lock_guard<std::mutex> lock(g_fsm_mutex);
        if (g_fsm_running.load() && g_fsm)
        {
            g_fsm->requestTransition(kPassiveStateId);
        }
        send(R"({"status":"estop_triggered"})");
    }
}

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);
    g_network_iface = vm["network"].as<std::string>();

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";
    std::cout.flush();

    send(R"({"status":"ready"})");

    // Read one command per line from stdin (this is what turns the process
    // into something websocketd can expose as a websocket endpoint).
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line == "connect")
        {
            cmd_connect();
        }
        else if (line == "run_policy")
        {
            cmd_run_policy();
        }
        else if (line == "exit_policy")
        {
            cmd_exit_policy();
        }
        else if (line == "estop")
        {
            cmd_estop();
        }
        else if (!line.empty())
        {
            send(R"({"status":"error","detail":"unknown_command"})");
        }
    }

    return 0;
}