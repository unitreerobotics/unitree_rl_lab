// ########### THIS IS OLD UNITREE CODE #######################

// // Copyright (c) 2025, Unitree Robotics Co., Ltd.
// // All rights reserved.

// #pragma once

// #include <unitree/common/thread/recurrent_thread.hpp>
// #include "BaseState.h"
// #include <spdlog/spdlog.h>
// #include <yaml-cpp/yaml.h>

// class CtrlFSM
// {
// public:
//     CtrlFSM(std::shared_ptr<BaseState> initstate)
//     {
//         // Initialize FSM states
//         states.push_back(std::move(initstate));

//     }

//     CtrlFSM(YAML::Node cfg)
//     {
//         auto fsms = cfg["_"]; // enabled FSMs

//         // register FSM string map; used for state transition
//         for (auto it = fsms.begin(); it != fsms.end(); ++it)
//         {
//             std::string fsm_name = it->first.as<std::string>();
//             int id = it->second["id"].as<int>();
//             FSMStringMap.insert({id, fsm_name});
//         }

//         // Initialize FSM states
//         for (auto it = fsms.begin(); it != fsms.end(); ++it)
//         {
//             std::string fsm_name = it->first.as<std::string>();
//             int id = it->second["id"].as<int>();
//             std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
//             auto fsm_class = getFsmMap().find("State_" + fsm_type);
//             if (fsm_class == getFsmMap().end()) {
//                 throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
//             }
//             auto state_instance = fsm_class->second(id, fsm_name);
//             add(state_instance);
//         }
//     }

//     void start() 
//     {
//         // Start From State_Passive
//         currentState = states[0];
//         currentState->enter();

//         fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
//             "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
//         spdlog::info("FSM: Start {}", currentState->getStateString());
//     }

//     void add(std::shared_ptr<BaseState> state)
//     {
//         for(auto & s : states)
//         {
//             if(s->isState(state->getState()))
//             {
//                 spdlog::error("FSM: State_{} already exists", state->getStateString());
//                 std::exit(0);
//             }
//         }

//         states.push_back(std::move(state));
//     }
    
//     ~CtrlFSM()
//     {
//         states.clear();
//     }

//     std::vector<std::shared_ptr<BaseState>> states;
// private:
//     const double dt = 0.001;

//     void run_()
//     {
//         currentState->pre_run();
//         currentState->run();
//         currentState->post_run();
        
//         // Check if need to change state
//         int nextStateMode = 0;
//         for(int i(0); i<currentState->registered_checks.size(); i++)
//         {
//             if(currentState->registered_checks[i].first())
//             {
//                 nextStateMode = currentState->registered_checks[i].second;
//                 break;
//             }
//         }

//         if(nextStateMode != 0 && !currentState->isState(nextStateMode))
//         {
//             for(auto & state : states)
//             {
//                 if(state->isState(nextStateMode))
//                 {
//                     spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
//                     currentState->exit();
//                     currentState = state;
//                     currentState->enter();
//                     break;
//                 }
//             }
//         }
//     }

//     std::shared_ptr<BaseState> currentState;
//     unitree::common::RecurrentThreadPtr fsm_thread_;
// };



















// Copyright (c) 2026, Muhammad Suleman.
// All rights reserved.

#pragma once

#include <atomic>
#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));
    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start()
    {
        // Start From State_Passive
        currentState = states[0];
        currentState->enter();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }

    ~CtrlFSM()
    {
        states.clear();
    }

    // Request a state transition from outside the FSM thread (e.g. a web GUI /
    // command-socket handler running on the main thread). Thread-safe.
    // Takes effect on the FSM's next 1ms tick and takes priority over the
    // current state's own registered_checks (joystick/keyboard-driven
    // transitions), same as pressing a mapped button would.
    void requestTransition(int stateId)
    {
        forced_transition_.store(stateId);
    }

    int getCurrentStateId() const
    {
        return currentState->getState();
    }

    std::string getCurrentStateString() const
    {
        return currentState->getStateString();
    }

    std::vector<std::shared_ptr<BaseState>> states;

private:
    const double dt = 0.001;
    std::atomic<int> forced_transition_{0};

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();

        // Check if need to change state.
        // A forced transition (requestTransition) always wins over the
        // state's own registered_checks.
        int nextStateMode = forced_transition_.exchange(0);

        if (nextStateMode == 0)
        {
            for(int i(0); i<currentState->registered_checks.size(); i++)
            {
                if(currentState->registered_checks[i].first())
                {
                    nextStateMode = currentState->registered_checks[i].second;
                    break;
                }
            }
        }

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            for(auto & state : states)
            {
                if(state->isState(nextStateMode))
                {
                    spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    break;
                }
            }
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};